#!/usr/bin/env python3
"""
Initialize a force field for fitting.

Takes a template force field, re-initializes angles and bonds via AAM on the
subset of the Sage optimizations that include Hessians, adds additional
torsion parameters from a JSON file, and finally resets all proper torsions
(including the new ones) to a sum of 4 cosines with force constants of zero.

This script relies on sage-2.2.1-optimizations.json, which is available at:
https://raw.githubusercontent.com/openforcefield/sage-2.2.1/refs/heads/main/02_curate-data/output/optimization-training-set.json

**Input files:**

sage-2.2.1-optimizations.json
    The collection of QCArchive Optimization records from which Hessians are
    selected for AAM initialization.

specific_torsions.jsonc
    JSON5 file including an array of objects with ``"label"`` and ``"smirks"``
    keys specifying proper torsions to add to the template force field.

openff_unconstrained-2.2.1.offxml
    Template force field. Can be collected automatically from openff-forcefields
    package, so does not need to be present in the folder. Saved to
    `intermediates/template.offxml` as a record.

**Output files:**

__main__.py.log
    Text log of processing.

intermediates/aam-singlepoint-hessians.json
    The collection of QCArchive Singlepoint records with Hessians used for AAM
    initialization.

intermediates/template.offxml

records/<record_id>/molecule.sdf
    SDFs of QCArchive records used for AAM initialization.

records/<record_id>/record.pickle
    Pickled QCArchive records used for AAM initialization.

aam-reset-specific.offxml
    The initialized force field.



**Notes: **
The Alice Allen Method is also known as the Modified Seminario Method (MSM),
but is here called AAM because calling a technique that has become
synonymous with initial parameter generation after a non-author of that
technique that gives the technique an initialism with another much more
widely used interpretation (Markov State Models) is frankly ridiculous.
"""

import pickle
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal, TypedDict, assert_never

import cyclopts
import json5
import numpy
from loguru import logger
from openff.qcsubmit.results.results import (
    BasicResultCollection,
    OptimizationResultCollection,
)
from openff.toolkit import ForceField, Molecule
from openff.toolkit.typing.engines.smirnoff import ProperTorsionHandler
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
    output_collection_path: Path = sibpath(
        "intermediates/aam-singlepoint-hessians.json",
    ),
    records_path: Path = sibpath("records"),
    ff_template_str: str = "openff_unconstrained-2.2.1.offxml",
    specific_torsions_path: Path = sibpath("specific_torsions.jsonc"),
    reset_torsion_periodicity: int = 4,
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
    specific_torsions_path
        Path to JSON file specifying the torsions to add to the force field.
        JSON should encode an array of objects with the keys ``"label"`` and
        ``"smirks"``, each being a string specifying the parameter ID and SMIRKS
        pattern respectively.
    reset_torsion_periodicity
        The periodicity to reset torsions to.
    output
        Path to generated force field
    """
    logger.info("---------------------- starting script ----------------------")
    collection = select_hessians(input_collection_path, output_collection_path)

    records_and_molecules = get_records(collection, records_path)

    aam_dataset = compute_aam_dataframe(records_and_molecules)

    ff_template = get_template_force_field(ff_template_str)

    force_field = initialize_bonds_and_angles(aam_dataset, ff_template)

    force_field = reset_and_split_torsions(
        force_field,
        specific_torsions_path,
        reset_torsion_periodicity,
    )

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

    collection_path_out.parent.mkdir(exist_ok=True, parents=True)
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
    for result in tqdm(
        list(flatten(collection.entries.values())),
        desc="loading from disk",
    ):
        record_path = records_path / f"{result.record_id}/record.pickle"
        molecule_path = records_path / f"{result.record_id}/molecule.sdf"

        with open(record_path, "rb") as f:
            record = pickle.load(f)
        molecule = Molecule.from_file(molecule_path, allow_undefined_stereo=True)
        assert isinstance(molecule, Molecule)
        records_and_molecules.append((record, molecule))

    return records_and_molecules


def get_template_force_field(ff_name: str) -> ForceField:
    """
    Get a force field that has the SMIRKS parameters we want to fill in with AAM
    """
    logger.info(f"Loading {ff_name}")
    ff = ForceField(ff_name, allow_cosmetic_attributes=True)

    template_output_path = Path(__file__).parent / "intermediates/template.offxml"
    template_output_path.parent.mkdir(exist_ok=True, parents=True)
    ff.to_file(str(template_output_path))
    logger.info(f"Loaded and saved to {template_output_path}'")
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
    """
    logger.info(
        "Computing initial bond and angle parameters via AAM (AKA MSM, see docstring)",
    )
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
    logger.info("Initial bond and angle parameters calculated")

    return DataFrame(entries)


def initialize_bonds_and_angles(
    aam_dataset: DataFrame,
    ff_template: ForceField,
    aggregator: Callable[[numpy.ndarray], float] = numpy.mean,
) -> ForceField:
    # Add the force field parameter ID corresponding to each harmonic to the dataframe
    logger.info("Labelling initial bond and angle parameters with force field IDs")
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
    logger.info("Initializing bond and angle force field parameters from AAM")
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
                        "Skipping equilibrium value update of linear angle"
                        + f" (AAM angle: {eq_agg.m_as('deg')} deg)",  # type: ignore
                    )
                    parameter.angle = eq_agg
            else:
                assert_never(parameter_type)

    return force_field


def reset_and_split_torsions(
    reference_force_field: ForceField,
    specific_torsions_json: Path,
    torsion_periodicity: int = 4,
) -> ForceField:
    """
    Add parameter splits that should re-use AAM parameters.
    """
    force_field = ForceField(reference_force_field.to_string())

    reference_torsion_handler = reference_force_field["ProperTorsions"]

    torsion_handler = ProperTorsionHandler(version="0.4")
    force_field.deregister_parameter_handler(force_field["ProperTorsions"])
    force_field.register_parameter_handler(torsion_handler)

    logger.info("Resetting reference torsions")
    for reference_torsion in reference_torsion_handler.parameters:
        new_parameter: dict[str, str | int | PlainQuantity[float] | float] = {
            "id": reference_torsion.id,
            "smirks": reference_torsion.smirks,
        }
        for i in range(torsion_periodicity):
            new_parameter[f"periodicity{i + 1}"] = i
            new_parameter[f"k{i + 1}"] = 0.0 * unit.kilocalories_per_mole
            new_parameter[f"phase{i + 1}"] = i % 2 * 180.0 * unit.degree
            new_parameter[f"idivf{i + 1}"] = 1.0

        torsion_handler.add_parameter(new_parameter)

    specific_torsion_types = json5.loads(
        specific_torsions_json.read_text(),
    )
    assert isinstance(specific_torsion_types, list)

    logger.info("Adding additional torsions")
    for torsion_type in specific_torsion_types:
        assert isinstance(torsion_type, dict)
        new_parameter: dict[str, str | int | PlainQuantity[float] | float] = {
            "id": torsion_type["label"],
            "smirks": torsion_type["smirks"],
        }
        for i in range(torsion_periodicity):
            new_parameter[f"periodicity{i + 1}"] = i
            new_parameter[f"k{i + 1}"] = 0.0 * unit.kilocalories_per_mole
            new_parameter[f"phase{i + 1}"] = i % 2 * 180.0 * unit.degree
            new_parameter[f"idivf{i + 1}"] = 1.0

        torsion_handler.add_parameter(new_parameter)

    return force_field


def write_smirnoff(ff: ForceField, filename: str | Path):
    """
    Write force field to disk.
    """
    logger.info(f"Writing initialized force field to {filename}")
    Path(filename).parent.mkdir(exist_ok=True, parents=True)
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
