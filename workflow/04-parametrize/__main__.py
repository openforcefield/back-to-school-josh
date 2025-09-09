#!/usr/bin/env python3
"""
TODO: Write me!

**Input files:**

File 1
    File 1 description

File 2
    File 2 description

**Output files:**

File 1
    File 1 description

File 2
    File 2 description
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import cyclopts
import datasets
import smee.converters
import torch
from loguru import logger
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule
from smee import TensorForceField, TensorTopology
from tqdm import tqdm

from back_to_school_josh.utils import sibpath


def main(
    *,
    output_tensor_ff_path: Path = sibpath("tensor_ff"),
    output_tensor_tops_path: Path = sibpath("tensor_tops"),
    dataset_paths: Mapping[str, Path] = {
        "spice2": sibpath("../02-select-data/datasets/spice2/train"),
        "tetramers": sibpath("../02-select-data/datasets/tetramers/train"),
    },
    force_field_paths: Sequence[Path] = (
        sibpath("../03-generate-initial-ff/aam-ff.offxml"),
    ),
    charge_model: Literal[
        "ambertools",
        "openeye",
        "nagl",
        "default_ff_charges",
    ] = "nagl",
):
    """
    Parametrize train sets into Smee/Descent tensor force field

    Parameters
    ----------
    output_tensor_ff_path
        Path to write Smee tensor force field to.
    output_tensor_tops_path
        Path to write Smee tensor topologies to.
    dataset_paths
        Paths to serialized Huggingface datasets to parametrize.
    force_field_paths
        Force field to parametrize with, expressed as a sequence of paths to
        OFFXML files.
    charge_model
        Charge model to parametrize with. ``"ambertools"``: AmberTools AM1BCC;
        ``"nagl"``: NAGL ``openff-gnn-am1bcc-0.1.0-rc.3.pt``; ``"openeye"``:
        OpenEye AM1BCC-ELF10; ``"default_ff_charges"``: Charges as specified in
        force field.
    """
    force_field = ForceField(*force_field_paths)

    # We will write the tensor force field to disk only once, and check that
    # all topologies index into identical force fields.
    shared_tensor_ff = None

    for _, path in dataset_paths.items():
        tensor_force_field, tensor_topologies = parametrize_dataset(
            path,
            force_field,
            charge_model=charge_model,
        )

        # Save this FF if it's the first one, otherwise check that its identical
        # to the saved one
        if shared_tensor_ff is None:
            shared_tensor_ff = tensor_force_field
            write_tensor_ff_to_disk(output_tensor_ff_path, shared_tensor_ff)
        else:
            assert tensor_force_field == shared_tensor_ff

        write_tensor_tops_to_disk(output_tensor_tops_path, tensor_topologies)


def parametrize_dataset(
    path: Path,
    force_field: ForceField,
    *,
    charge_model: Literal["ambertools", "openeye", "nagl", "default_ff_charges"],
) -> tuple[TensorForceField, dict[str, TensorTopology]]:
    logger.info(f"Loading dataset at {path}...")
    dataset = datasets.Dataset.load_from_disk(str(path))
    logger.info("Loaded.")

    logger.info(f"Constructing interchanges with {force_field}...")
    # TODO: Batching and multiprocessing!
    interchanges = [
        smiles_to_interchange(smiles, force_field, charge_model=charge_model)
        for smiles in tqdm(dataset["smiles"], desc="Parametrizing")
    ]
    logger.info("Interchanges constructed.")

    logger.info("Converting to tensor format...")
    tensor_force_field, tensor_topologies = smee.converters.convert_interchange(
        interchanges,
    )
    logger.info("Converted.")

    tensor_topology_dict = {
        smiles: tensor_topology
        for smiles, tensor_topology in zip(
            dataset["smiles"],
            tensor_topologies,
            strict=True,
        )
    }

    return tensor_force_field, tensor_topology_dict


def smiles_to_interchange(
    smiles: str,
    force_field: ForceField,
    *,
    charge_model: Literal[
        "default_ff_charges",
        "ambertools",
        "openeye",
        "nagl",
    ] = "nagl",
) -> Interchange:
    molecule = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)

    # Assign partial charges with requested charge model
    match charge_model:
        case "default_ff_charges":
            charge_from_molecules = None
        case "ambertools":
            from openff.toolkit import AmberToolsToolkitWrapper

            molecule.assign_partial_charges(
                partial_charge_method="am1bcc",
                toolkit_registry=AmberToolsToolkitWrapper(),
            )
            charge_from_molecules = [molecule]
        case "openeye":
            from openff.toolkit import OpenEyeToolkitWrapper

            molecule.assign_partial_charges(
                partial_charge_method="am1bccelf10",
                toolkit_registry=OpenEyeToolkitWrapper(),
            )
            charge_from_molecules = [molecule]
        case "nagl":
            from openff.toolkit import NAGLToolkitWrapper

            molecule.assign_partial_charges(
                partial_charge_method="openff-gnn-am1bcc-0.1.0-rc.3.pt",
                toolkit_registry=NAGLToolkitWrapper(),
            )
            charge_from_molecules = [molecule]

    # Construct interchange
    return Interchange.from_smirnoff(
        force_field,
        [molecule],
        charge_from_molecules=charge_from_molecules,
    )


def write_tensor_ff_to_disk(
    output_tensor_ff_path: Path,
    shared_tensor_ff: TensorForceField,
):
    output_tensor_ff_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(shared_tensor_ff, output_tensor_ff_path)


def write_tensor_tops_to_disk(
    output_tensor_tops_path: Path,
    tensor_topologies: Mapping[str, TensorTopology],
):
    output_tensor_tops_path.parent.mkdir(exist_ok=True, parents=True)
    for smiles, ttop in tensor_topologies.items():
        torch.save(ttop, output_tensor_tops_path / smiles)


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
