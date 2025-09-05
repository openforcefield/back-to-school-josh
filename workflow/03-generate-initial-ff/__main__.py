#!/usr/bin/env python3
"""
TODO: Write me!

This script relies on sage-2.2.1-optimizations.json, which is available at:
https://raw.githubusercontent.com/openforcefield/sage-2.2.1/refs/heads/main/02_curate-data/output/optimization-training-set.json

**Output files:**

file1
    Description of file1

file2
    Description of file2
"""

import pickle
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal, TypedDict, assert_never

import cyclopts
import numpy
from loguru import logger
from openff.qcsubmit.results.results import (
    BasicResultCollection,
    OptimizationResultCollection,
)
from openff.toolkit import ForceField, Molecule
from openff.toolkit.typing.engines.smirnoff.parameters import ValenceDict
from openff.units import unit
from pandas import DataFrame
from pint.facets.plain import PlainQuantity
from qcportal.client import SinglepointDriver, SinglepointRecord
from qubekit.bonded.mod_seminario import ModSeminario
from qubekit.forcefield import HarmonicBondParameter
from qubekit.forcefield.parameters import HarmonicAngleParameter
from qubekit.molecules import Ligand
from tqdm import tqdm

from back_to_school_josh.qcsubmit import CustomResultFilter
from back_to_school_josh.utils import flatten, sibpath, unwrap


def main(
    *,
    input_collection_path: Path = sibpath("sage-2.2.1-optimizations.json"),
    output_collection_path: Path = sibpath("aam-singlepoint-hessians.json"),
    records_path: Path = sibpath("records"),
    ff_template_str: str = "openff_unconstrained-2.2.1.offxml",
    output: Path = sibpath("aam-ff.offxml"),
):
    """
    Compute an initial force field from a QCSubmit collection via MSM/AAM.

    Parameters
    ----------
    input_collection_path
        Path to JSON-serialized optimization collection to be filtered.
    output_collection_path
        Path to filtered JSON-serialized singlepoint collection. Note that if
        this file exists, it will be loaded in lieu of filtering the input file
        as an optimization.
    records_path
        Path to downloaded QCArchive records.
    ff_template_str
        Force field to generate angle and bond parameter values for. This can be
        a path relative to the working directory to an OFFXML file, the name of
        a published OFFXML file, or the contents of an OFFXML file.
    output
        Path to generated force field
    """
    collection = select_hessians(input_collection_path, output_collection_path)

    records_and_molecules = get_records(collection, records_path)

    aam_dataset = compute_aam_dataframe(records_and_molecules)

    ff_template = get_parameter_smirks(ff_template_str)

    force_field = initialize_force_field(aam_dataset, ff_template)

    force_field = split_params_after_aam(force_field)

    write_smirnoff(force_field, output)


def select_hessians(
    collection_path_in: Path,
    collection_path_out: Path,
) -> BasicResultCollection:
    """
    Select a dataset of hessians from QCArchive.
    """
    if collection_path_out.is_file():
        logger.info("Loading filtered result collection...")
        filtered = BasicResultCollection.parse_file(collection_path_out)
        logger.info("Loaded.")
        logger.warning(
            "Filtered result collection loaded from cache without"
            + " checking it agrees with input result collection!",
        )

        return filtered

    from openff.qcsubmit.results.filters import LowestEnergyFilter

    logger.info("Loading optimization result collection...")
    dataset = OptimizationResultCollection.parse_file(collection_path_in)
    logger.info("Loaded.")

    logger.info("Filtering on energy...")
    filtered = dataset.filter(LowestEnergyFilter())

    logger.info("Filtering on hessians...")
    filtered = filtered.to_basic_result_collection(driver="hessian")

    logger.info("Filtered.")

    collection_path_out.write_text(filtered.json())

    return filtered


def get_records(
    collection: BasicResultCollection,
    records_path: Path,
) -> Sequence[tuple[SinglepointRecord, Molecule]]:
    """Load the records in the collection from disk, or download them if missing"""
    logger.info("Finding missing records")
    # Download all records that haven't already been downloaded
    missing_results = collection.filter(
        CustomResultFilter.from_filter_function(
            lambda result: not (
                Path.is_file(
                    records_path / f"{result.record_id}/record.pickle",
                )
                and Path.is_file(
                    records_path / f"{result.record_id}/molecule.sdf",
                )
            ),
        ),
    )
    logger.info(
        f"Downloading {len(list(flatten(missing_results.entries.values())))} missing records",
    )
    downloaded_records = missing_results.to_records()

    # Save all newly downloaded records to disk
    logger.info("Saving to disk")
    for record, molecule in downloaded_records:
        record_path = records_path / f"{record.id}/record.pickle"
        molecule_path = records_path / f"{record.id}/molecule.sdf"

        record_path.parent.mkdir(parents=True, exist_ok=True)
        with open(record_path, "wb") as file:
            pickle.dump(record, file, protocol=pickle.HIGHEST_PROTOCOL)

        molecule.to_file(molecule_path, "SDF")

    # Load all records in the collection from disk (they should all be there now!)
    logger.info("Loading desired records from disk")
    records_and_molecules: list[tuple[SinglepointRecord, Molecule]] = []
    for result in tqdm(flatten(collection.entries.values()), desc="loading from disk"):
        record_path = records_path / f"{result.record_id}/record.pickle"
        molecule_path = records_path / f"{result.record_id}/molecule.sdf"

        with open(record_path, "rb") as f:
            record = pickle.load(f)
        molecule = Molecule.from_file(molecule_path, allow_undefined_stereo=True)
        assert isinstance(molecule, Molecule)
        records_and_molecules.append((record, molecule))

    return records_and_molecules


def get_parameter_smirks(ff_name: str) -> ForceField:
    """
    Get a force field that has the SMIRKS parameters we want to fill in with AAM
    """
    ff = ForceField(ff_name, allow_cosmetic_attributes=True)
    ff.to_file(str(Path(__file__).parent / "template.offxml"))
    return ff


def compute_aam_dataframe(
    records_and_molecules: Iterable[tuple[SinglepointRecord, Molecule]],
) -> DataFrame:
    """
    Compute equilibrium values and force constants with the Alice Allen Method.

    Returns
    -------
    aam_df
        A dataframe with one row per bond or angle in each record and the
        following columns:
         - ``"mapped_smiles"``: A mapped isomeric explicit hydrogen SMILES for
         the record's molecule
         - ``"id"``: The record's ID
         - ``"parameter_type"``: Either ``"Bonds"`` or ``"Angles"``
         - ``"indices"``: The indices into the ``Molecule`` produced from the
         CMILES that correspond to this bond or angle
         - ``"eq"``: The equilibrium value of the bond or angle
         - ``"k"``: The force constant of the bond or angle

    Notes
    -----
    The Alice Allen Method is also known as the Modified Seminario Method (MSM),
    but is here called AAM because calling a technique that has become
    synonymous with initial parameter generation after a non-author of that
    technique that gives the technique an initialism with another much more
    widely used interpretation (Markov State Models) is frankly ridiculous.
    """
    aam = ModSeminario(vibrational_scaling=1.0)

    entries: list[AamBondEntry | AamAngleEntry] = []
    for record, molecule in tqdm(
        records_and_molecules,
        desc="Computing AAM harmonics for records",
    ):
        qube_mol = to_qubemol(molecule, record)
        if qube_mol.hessian is None:
            logger.warning(
                f"{molecule.to_smiles(explicit_hydrogens=False)} (record"
                + f" {record.id}) has no hessians! Skipping...",
            )
            continue
        qube_mol = aam.run(qube_mol)
        entries.extend(extract_qubemol_parameters(qube_mol))

    return DataFrame(entries)


def initialize_force_field(
    aam_dataset: DataFrame,
    ff_template: ForceField,
    aggregator: Callable[[numpy.ndarray], float] = numpy.mean,
) -> ForceField:
    # Add the force field parameter ID corresponding to each harmonic to the dataframe
    labels: dict[str, dict[str, ValenceDict]] = {
        mapped_smiles: unwrap(
            ff_template.label_molecules(
                Molecule.from_mapped_smiles(
                    mapped_smiles,
                    allow_undefined_stereo=True,
                ).to_topology(),
            ),
        )
        for mapped_smiles in tqdm(
            aam_dataset["mapped_smiles"].unique(),
            desc="labelling molecules",
        )
    }
    aam_dataset["parameter_id"] = [
        labels[mapped_smiles][parameter_type][indices].id
        for mapped_smiles, parameter_type, indices in zip(
            aam_dataset["mapped_smiles"],
            aam_dataset["parameter_type"],
            aam_dataset["indices"],
        )
    ]

    force_field = ForceField(ff_template.to_string())

    # Aggregate and update the force field parameters!
    for parameter_type, parameter_type_df in aam_dataset.groupby("parameter_type"):
        assert parameter_type in ("Bonds", "Angles"), parameter_type
        handler = force_field.get_parameter_handler(parameter_type)

        for parameter_id, parameter_df in tqdm(
            parameter_type_df.groupby("parameter_id"),
            desc=f"Aggregating parameters for {parameter_type}",
        ):
            assert isinstance(parameter_id, str), parameter_id
            parameter = unwrap(handler.get_parameter({"id": parameter_id}))
            k_agg = aggregator(
                unit.Quantity.from_sequence(
                    numpy.asarray(parameter_df["k"]),  # type: ignore
                ),
            )
            eq_agg = aggregator(
                unit.Quantity.from_sequence(
                    numpy.asarray(parameter_df["eq"]),  # type: ignore
                ),
            )

            if parameter_type == "Bonds":
                parameter.length = eq_agg
                parameter.k = k_agg
            elif parameter_type == "Angles":
                parameter.k = k_agg
                # Only update the parameter if it isn't linear
                if parameter.angle.m_as(unit.degree) == 180.0:
                    logger.info(
                        f"Skipping equilibrium value update of linear angle (AAM angle: {eq_agg})",
                    )
                    parameter.angle = eq_agg
            else:
                assert_never(parameter_type)

    return force_field


def split_params_after_aam(force_field: ForceField) -> ForceField:
    """
    Add parameter splits that should re-use AAM parameters.
    """
    raise NotImplementedError()


def write_smirnoff(ff: ForceField, filename: str | Path):
    """
    Write force field to disk.
    """
    ff.to_file(str(filename))


def extract_qubemol_parameters(
    qube_mol: Ligand,
) -> Iterator["AamBondEntry | AamAngleEntry"]:
    for bond_parameter in qube_mol.BondForce:
        assert isinstance(bond_parameter, HarmonicBondParameter)
        a, b = bond_parameter.atoms
        assert isinstance(a, int)
        assert isinstance(b, int)
        if a < b:
            indices = (a, b)
        else:
            indices = (b, a)

        yield AamBondEntry(
            {
                "mapped_smiles": qube_mol.provenance["mapped_smiles"],
                "id": qube_mol.provenance["qcarchive_record_id"],
                "parameter_type": "Bonds",
                "indices": indices,
                "eq": unit.Quantity(bond_parameter.length, "nm"),
                "k": unit.Quantity(bond_parameter.k, "kJ/mol/nm**2"),
            },
        )

    for angle_parameter in qube_mol.AngleForce:
        assert isinstance(angle_parameter, HarmonicAngleParameter)
        a, b, c = angle_parameter.atoms
        assert isinstance(a, int)
        assert isinstance(b, int)
        assert isinstance(c, int)
        if a < c:
            indices = (a, b, c)
        else:
            indices = (c, b, a)

        yield AamAngleEntry(
            mapped_smiles=qube_mol.provenance["mapped_smiles"],
            id=qube_mol.provenance["qcarchive_record_id"],
            parameter_type="Angles",
            indices=indices,
            eq=unit.Quantity(angle_parameter.angle, "rad"),
            k=unit.Quantity(angle_parameter.k, "kJ/mol/rad**2"),
        )


def to_qubemol(offmol: Molecule, record: SinglepointRecord) -> Ligand:
    offmol.add_conformer(
        numpy.array(record.molecule.geometry, float).reshape(-1, 3) * unit.bohr,
    )
    qube_mol = Ligand.from_rdkit(offmol.to_rdkit(), name=offmol.name or "offmol")
    qube_mol.provenance["mapped_smiles"] = offmol.to_smiles(
        isomeric=True,
        explicit_hydrogens=True,
        mapped=True,
    )
    qube_mol.provenance["qcarchive_record_id"] = record.id
    assert record.specification.driver == SinglepointDriver.hessian
    qube_mol.hessian = record.return_result
    return qube_mol


class AamEntry(TypedDict):
    mapped_smiles: str
    id: int
    eq: PlainQuantity[float]
    k: PlainQuantity[float]


class AamBondEntry(AamEntry):
    parameter_type: Literal["Bonds"]
    indices: tuple[int, int]


class AamAngleEntry(AamEntry):
    parameter_type: Literal["Angles"]
    indices: tuple[int, int, int]


if __name__ == "__main__":
    logger.add(Path(__file__).with_suffix(".py.log"), delay=True)

    app = cyclopts.App(
        name=(
            Path(__file__).parent.stem
            if Path(__file__).name == "__main__.py"
            else Path(__file__).stem
        ),
        help=__doc__,
        help_format="restructuredtext",
    )
    app.default()(main)
    app()
