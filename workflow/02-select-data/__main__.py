import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import datasets
import descent.targets.energy
import h5py
import numpy
import typer
from openff.units import Quantity, unit
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from torch import Tensor
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
    spice_file: Annotated[
        Path,
        typer.Argument(help="Path to find SPICE data file"),
    ] = Path(".data/SPICE-2.0.1.hdf5"),
    tetramers_files: Annotated[
        Path,
        typer.Argument(help="Path to find downloaded QCArchive data files"),
    ] = Path(".data/qcarchive/records"),
    output_dir: Annotated[
        Path,
        typer.Argument(help="Path to find save filter"),
    ] = Path("workflow/02-select-data/datasets"),
    spice_subsets: Annotated[
        list[str] | None,
        typer.Argument(
            help="Names of the subsets in the spice file to use",
            show_default=" ".join(repr(s) for s in DEFAULT_SPICE_SUBSETS),
        ),
    ] = None,
):
    if spice_subsets is None:
        spice_subsets = DEFAULT_SPICE_SUBSETS

    spice_ds = load_spice(spice_file, subsets=spice_subsets)
    tetramers_ds = load_qcarchive(tetramers_files)

    spice_ds = filter_forces(spice_ds)

    train_spice_ds, test_spice_ds = train_test_split(
        spice_ds,
        frac_test=0.05,
    )
    train_tetramers_ds, test_tetramers_ds = train_test_split(
        tetramers_ds,
        frac_test=0.05,
    )

    train_spice_ds.save_to_disk(output_dir / "spice2/train")
    test_spice_ds.save_to_disk(output_dir / "spice2/test")
    train_tetramers_ds.save_to_disk(output_dir / "tetramers/train")
    test_tetramers_ds.save_to_disk(output_dir / "tetramers/train")


def load_spice(spice_file: Path, subsets: Sequence[str]) -> datasets.Dataset:
    data: list[descent.targets.energy.Entry] = []

    with h5py.File(spice_file) as hdf5_file:
        for record in tqdm(
            hdf5_file.values(),
            desc=f"Extracting {spice_file.name}",
            ncols=80,
        ):
            assert isinstance(record, h5py.Group), type(record)

            smiles = record["smiles"].asstr()[0]  # type: ignore
            assert isinstance(smiles, str), type(smiles)

            subset = record["subset"].asstr()[0]  # type: ignore
            assert isinstance(subset, str), type(subset)

            if subset not in subsets:
                continue

            conformations = (
                Quantity(
                    numpy.asarray(record["conformations"]),
                    "hartree",
                )
                * unit.avogadro_constant
            ).m_as("kcal/mol")
            assert isinstance(conformations, numpy.ndarray), type(conformations)

            dft_total_energy = Quantity(
                numpy.asarray(record["dft_total_energy"]),
                "bohr",
            ).m_as("angstrom")
            assert isinstance(dft_total_energy, numpy.ndarray), type(dft_total_energy)

            dft_total_gradient = (
                Quantity(
                    numpy.asarray(record["dft_total_gradient"]),
                    "hartree/bohr",
                )
                * unit.avogadro_constant
            ).m_as("kcal/mol/angstrom")
            assert isinstance(dft_total_gradient, numpy.ndarray), type(
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
                    forces=Tensor(dft_total_gradient),
                ),
            )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ds = descent.targets.energy.create_dataset(data)
    return ds


def load_qcarchive(files: Path) -> datasets.Dataset:
    raise NotImplementedError()


def filter_forces(ds: datasets.Dataset) -> datasets.Dataset:
    raise NotImplementedError()


def train_test_split(
    ds: datasets.Dataset,
    frac_test: float,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    assert 0.0 < frac_test < 1.0

    n_smiles = len(ds["smiles"])
    n_test = round(frac_test * n_smiles)

    train_indices = choose_diverse_molecules(n_test, ds["smiles"])
    train_indices_set = train_indices
    assert len(train_indices) == len(train_indices_set)
    test_indices = [i for i in range(n_smiles) if i not in train_indices_set]

    train_ds = ds.select(indices=train_indices)
    test_ds = ds.select(indices=test_indices)
    return train_ds, test_ds


def choose_diverse_molecules(
    n: int,
    smiles: Sequence[str],
    seed: int | None = None,
) -> Sequence[int]:
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


if __name__ == "__main__":
    app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
    app.command(help=__doc__)(main)
    app()
