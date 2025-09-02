#!/usr/bin/env python3
"""
Filter, split, and convert the 4mers and SPICE2 datasets into Descent datasets.

Output files:
    datasets/spice2/test
        Descent/huggingface test dataset for SPICE2
    datasets/spice2/train
        Descent/huggingface train dataset for SPICE2
    datasets/tetramers/test
        Descent/huggingface test dataset for 4-mers
    datasets/tetramers/train
        Descent/huggingface train dataset for 4-mers
    __main__.py.forces.png
        Histogram of max absolute forces and filtered cutoffs
"""

import pickle
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, assert_never

import datasets
import descent.targets.energy
import h5py
import matplotlib.pyplot as plt
import numpy as np
import typer
from loguru import logger
from openff.units import Quantity, unit
from qcportal.client import SinglepointRecord
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from torch import Tensor, tensor
from tqdm import tqdm

DEFAULT_SPICE_SUBSETS = [
    "SPICE DES Monomers Single Points Dataset v1.1",
    "SPICE Dipeptides Single Points Dataset v1.3",
    "SPICE PubChem Set 1 Single Points Dataset v1.3",
    "SPICE PubChem Set 2 Single Points Dataset v1.3",
    "SPICE PubChem Set 3 Single Points Dataset v1.3",
    "SPICE PubChem Set 4 Single Points Dataset v1.3",
    "SPICE PubChem Set 5 Single Points Dataset v1.3",
    "SPICE PubChem Set 6 Single Points Dataset v1.3",
    "SPICE PubChem Set 7 Single Points Dataset v1.0",
    "SPICE PubChem Set 8 Single Points Dataset v1.0",
    "SPICE PubChem Set 9 Single Points Dataset v1.0",
    "SPICE PubChem Set 10 Single Points Dataset v1.0",
]


def main(
    *,
    spice_file: Annotated[
        Path,
        typer.Option(help="Path to find SPICE data file"),
    ] = Path(".data/SPICE-2.0.1.hdf5"),
    tetramers_dir: Annotated[
        Path,
        typer.Option(help="Path to find downloaded QCArchive records"),
    ] = Path(".data/qcarchive/"),
    output_dir: Annotated[
        Path,
        typer.Option(help="Path to find save filter"),
    ] = Path("workflow/02-select-data/datasets"),
    spice_subsets: Annotated[
        list[str],
        typer.Option(help="Names of the subsets in the spice file to use"),
    ] = DEFAULT_SPICE_SUBSETS,
    spice_n_records: Annotated[
        int | None,
        typer.Option(
            help="Load only the first this many records from SPICE",
            show_default="All",
        ),
    ] = None,
    spice_drop_forces: Annotated[
        float,
        typer.Option(
            help="Drop this proportion of all SPICE entries with the greatest forces",
        ),
    ] = 0.025,
    spice_drop_forces_metric: Annotated[
        str,
        typer.Option(
            help=(
                "The metric to use when filtering forces."
                + " MABS: Maximum absolute force. RMS: Root mean square force."
            ),
        ),
    ] = "MABS",
):
    spice_ds = load_spice(
        spice_file,
        subsets=spice_subsets,
        stop_early=spice_n_records,
    )

    spice_ds = filter_forces(
        spice_ds,
        keep_frac=1.0 - spice_drop_forces,
        metric=spice_drop_forces_metric,
    )

    logger.info("Splitting SPICE2 dataset into train and test")
    train_spice_ds, test_spice_ds = train_test_split(
        spice_ds,
        frac_test=0.05,
    )

    logger.info("Saving SPICE2 datasets to disk")
    train_spice_ds.save_to_disk(output_dir / "spice2/train")
    test_spice_ds.save_to_disk(output_dir / "spice2/test")

    tetramers_ds = load_qcarchive(tetramers_dir.glob("records/*/record.pickle"))

    logger.info("Splitting tetramers dataset into train and test")
    train_tetramers_ds, test_tetramers_ds = train_test_split(
        tetramers_ds,
        frac_test=0.05,
    )

    logger.info("Saving tetramers datasets to disk")
    train_tetramers_ds.save_to_disk(output_dir / "tetramers/train")
    test_tetramers_ds.save_to_disk(output_dir / "tetramers/test")


def load_spice(
    spice_file: Path,
    subsets: Sequence[str],
    stop_early: int | None = None,
) -> datasets.Dataset:
    """
    Load certain subsets of a SPICE dataset from a hdf5 file into a Smee dataset.
    """
    logger.info(f"Loading {subsets} from {spice_file}")
    data: list[descent.targets.energy.Entry] = []

    with h5py.File(spice_file) as hdf5_file:
        for i, record in enumerate(
            tqdm(
                hdf5_file.values(),
                desc=f"Extracting {spice_file.name}",
                ncols=80,
            ),
        ):
            if stop_early is not None and i >= stop_early:
                logger.info(f"Stopping after {i} records as requested")
                break

            assert isinstance(record, h5py.Group), type(record)

            smiles = record["smiles"].asstr()[0]  # type: ignore
            assert isinstance(smiles, str), type(smiles)

            subset = record["subset"].asstr()[0]  # type: ignore
            assert isinstance(subset, str), type(subset)

            if subset not in subsets:
                continue

            conformations = Quantity(
                np.asarray(record["conformations"]),
                "bohr",
            ).m_as("angstrom")
            assert isinstance(conformations, np.ndarray), type(conformations)

            dft_total_energy = (
                Quantity(
                    np.asarray(record["dft_total_energy"]),
                    "hartree",
                )
                * unit.avogadro_constant
            ).m_as("kcal/mol")
            assert isinstance(dft_total_energy, np.ndarray), type(dft_total_energy)

            dft_total_gradient = (
                Quantity(
                    np.asarray(record["dft_total_gradient"]),
                    "hartree/bohr",
                )
                * unit.avogadro_constant
            ).m_as("kcal/mol/angstrom")
            assert isinstance(dft_total_gradient, np.ndarray), type(
                dft_total_gradient,
            )

            n_conformers = conformations.shape[0]
            n_atoms = conformations.shape[1]

            assert conformations.shape == (n_conformers, n_atoms, 3), (
                f"expected {(n_conformers, n_atoms, 3)}, got {conformations.shape} for {smiles}"
            )
            assert dft_total_energy.shape == (n_conformers,), (
                f"expected {(n_conformers,)}, got {dft_total_energy.shape} for {smiles}"
            )
            assert dft_total_gradient.shape == (n_conformers, n_atoms, 3), (
                f"expected {(n_conformers, n_atoms, 3)}, got {dft_total_gradient.shape} for {smiles}"
            )

            data.append(
                descent.targets.energy.Entry(
                    smiles=smiles,
                    coords=Tensor(conformations),
                    energy=Tensor(dft_total_energy),
                    forces=Tensor(-dft_total_gradient),
                ),
            )
    logger.info("Constructing dataset...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ds = descent.targets.energy.create_dataset(data)
    logger.info(f"Dataset constructed with {len(ds)} molecules")
    return ds


def load_qcarchive(record_paths: Iterable[Path]) -> datasets.Dataset:
    """Load pickled qcarchive single point records into a Smee dataset"""
    logger.info("Loading records...")
    records = [
        unpickle(path)
        for path in tqdm(
            record_paths,
            desc="Unpickling records",
            ncols=80,
        )
    ]
    logger.info(f"Loaded {len(records)} records")

    data: list[descent.targets.energy.Entry] = []
    for record in tqdm(
        records,
        desc="Converting to Smee dataset entries",
        ncols=80,
    ):
        assert isinstance(record, SinglepointRecord)

        last = record
        last_mol = last.molecule
        assert last_mol.identifiers is not None
        assert last.properties is not None

        smiles = last_mol.identifiers.canonical_isomeric_explicit_hydrogen_mapped_smiles
        assert isinstance(smiles, str)

        coords = Quantity(last_mol.geometry, "bohr").m_as("angstrom")
        energy = (
            Quantity(last.properties["return_energy"], "hartree")
            * unit.avogadro_constant
        ).m_as(
            "kcal/mol",
        )
        gradient = np.array(last.properties["scf total gradient"]).reshape((-1, 3))
        forces = (Quantity(-gradient, "hartree/bohr") * unit.avogadro_constant).m_as(
            "kcal/mol/angstrom",
        )
        assert isinstance(forces, np.ndarray)

        data.append(
            descent.targets.energy.Entry(
                smiles=smiles,
                coords=tensor(coords),
                energy=tensor([energy]),
                forces=tensor(forces),
            ),
        )

    logger.info("Constructing dataset...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ds = descent.targets.energy.create_dataset(data)

    logger.info(f"Dataset constructed with {len(ds)} molecules")
    return ds


def filter_forces(
    ds: datasets.Dataset,
    keep_frac: float,
    metric: Literal["RMS", "MABS"] = "MABS",
) -> datasets.Dataset:
    """Remove entries with high forces

    Parameters
    ==========
    ds
        The dataset to filter
    keep_frac
        The fraction of dataset entries to keep
    metric
        The metric to use to filter forces
    """
    assert keep_frac <= 1.0

    if metric == "RMS":
        forces = tensor([tensor.square().mean().sqrt() for tensor in ds["forces"]])
    elif metric == "MABS":
        forces = tensor([tensor.absolute().max() for tensor in ds["forces"]])
    else:
        assert_never(metric)

    assert forces.shape == (len(ds),), f"expected {(len(ds),)} got {forces.shape}"

    fig = plt.figure()
    ax = fig.subplots()
    ax.hist(forces, log=True, bins="fd")
    percentiles = [keep_frac * 100, 90, 95, 99]
    for percentile_force, percentile, linestyle in zip(
        np.percentile(forces.numpy(), percentiles),
        percentiles,
        ["-", "--", "-.", ":"],
    ):
        ax.axvline(
            percentile_force,
            label=f"{percentile}th percentile",
            color="black",
            linestyle=linestyle,
        )
    ax.set_xlabel(f"{metric} Force (kcal/mol/angstrom)")
    ax.set_ylabel("Frequency")
    ax.set_xlim(xmin=forces.min().item(), xmax=forces.max().item())
    ax.legend()
    fig.savefig(Path(__file__).with_suffix(".py.forces.png"))

    indices = forces.argsort()[: round(keep_frac * len(forces))]
    logger.info(
        f"Keeping {len(indices)}/{len(ds)} entries ({len(indices) / len(ds) * 100}%).",
    )
    logger.info(f"Largest retained {metric} force: {forces[indices[-1]]} kcal/mol/Å")

    print(indices, indices.sort())
    return ds.select(indices)


def train_test_split(
    ds: datasets.Dataset,
    frac_test: float,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    """Split ds into train and test subsets with diverse molecules in train

    Returns
    =======
    (train_ds, test_ds)
        train and test datasets
    """
    assert 0.0 < frac_test < 1.0

    n_smiles = len(ds["smiles"])
    n_test = round(frac_test * n_smiles)

    test_indices = choose_diverse_molecules(n_test, ds["smiles"])
    test_indices_set = test_indices
    assert len(test_indices) == len(test_indices_set)
    train_indices = [i for i in range(n_smiles) if i not in test_indices_set]

    train_ds = ds.select(indices=train_indices)
    test_ds = ds.select(indices=test_indices)
    assert len(train_ds) + len(test_ds) == len(ds)
    logger.info(
        f"split dataset of {len(ds)} molecules into"
        + f" training set of {len(train_ds)} and testing set of {len(test_ds)}",
    )
    return train_ds, test_ds


def choose_diverse_molecules(
    n: int,
    smiles: Sequence[str],
    seed: int | None = None,
) -> Sequence[int]:
    """Choose n diverse molecules from a sequence of SMILES"""
    fingerprinter = rdFingerprintGenerator.GetMorganGenerator(radius=3)
    fingerprints = [
        fingerprinter.GetFingerprint(Chem.MolFromSmiles(s))
        for s in tqdm(
            smiles,
            desc="Computing fingerprints",
            ncols=80,
        )
    ]

    picker = MaxMinPicker()
    return picker.LazyBitVectorPick(
        fingerprints,
        len(fingerprints),
        n,
        seed=-1 if seed is None else seed,
    )


def unpickle(path: Path) -> Any:
    """Unpickle the file at path"""
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    logger.add(Path(__file__).with_suffix(".py.log"), delay=True)
    app = typer.Typer(
        add_completion=False,
        rich_markup_mode="rich",
        pretty_exceptions_enable=False,
    )
    app.command(help=__doc__)(main)
    app()
