#!/usr/bin/env python3
"""
Parametrize all molecules in all datasets with the given force field and charge model.


**Output files:**


outputs/tensor_ff.pt
    Pickled Smee tensor force field indexed into by all topologies. Load with
    ``pytorch.load(path, weights_only=False)``.

outputs/smiles_to_topologies.pt
    Pickled dictionary from mapped SMILES strings used in datasets to the
    corresponding Smee tensor topology. All topologies index into the output
    tensorforce field. Load with ``pytorch.load(path, weights_only=False)``.

04-parametrize.py.log
    Text log of all operations.
"""

import functools
import multiprocessing
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import cyclopts
import datasets
import smee.converters
import torch
from loguru import Logger, logger
from openff.interchange import Interchange
from openff.toolkit import ForceField, Molecule
from smee import TensorForceField, TensorTopology
from tqdm import tqdm

from back_to_school_josh.utils import sibpath


def main(
    *,
    output_tensor_ff_path: Path = sibpath("outputs/tensor_ff.pt"),
    output_tensor_tops_path: Path = sibpath("outputs/smiles_to_topologies.pt"),
    dataset_paths: dict[str, Path] = {
        "spice2_train": sibpath("../02-select-data/datasets/spice2/train"),
        "spice2_test": sibpath("../02-select-data/datasets/spice2/test"),
        "tetramers_train": sibpath("../02-select-data/datasets/tetramers/train"),
        "tetramers_test": sibpath("../02-select-data/datasets/tetramers/test"),
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
    n_processes: int | None = None,
):
    """
    Parametrize train sets into Smee/Descent tensor force field

    Parameters
    ----------
    output_tensor_ff_path
        Path of file to write Smee tensor force field to.
    output_tensor_tops_path
        Path of directory to write Smee tensor topologies to.
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
    n_processes
        Number of parallel processes to use to parametrize systems. If ``None``
        or not specified, use all available cores.
    """
    logger.info("---------------------- starting script ----------------------")
    force_field = ForceField(*force_field_paths)

    smiles_to_interchange_map: dict[str, Interchange] = {}
    for _, ds_path in dataset_paths.items():
        smiles_to_interchange_map.update(
            parametrize_dataset(
                ds_path,
                force_field,
                charge_model=charge_model,
                n_processes=n_processes,
            ),
        )

    tensor_force_field, tensor_topologies = interchanges_to_smee(
        smiles_to_interchange_map,
    )

    write_tensor_ff_to_disk(
        output_tensor_ff_path,
        tensor_force_field,
    )

    # Finally, write the tensor topologies to disk
    write_tensor_tops_to_disk(output_tensor_tops_path, tensor_topologies)


def parametrize_dataset(
    path: Path,
    force_field: ForceField,
    *,
    charge_model: Literal["ambertools", "openeye", "nagl", "default_ff_charges"],
    n_processes: int | None = None,
) -> dict[str, Interchange]:
    """
    Parametrize all molecules in the dataset by applying ``force_field``.

    Parameters
    ----------
    path
        Path to a Huggingface dataset directory with a ``"smiles"`` column
        containing mapped SMILES strings. The molecules represented by these
        SMILES strings will be parametrized.
    force_field
        A SMIRNOFF Force Field object. Used to parametrize the molecules at
        ``path``.
    charge_model
        Charge model to parametrize with. ``"ambertools"``: AmberTools AM1BCC;
        ``"nagl"``: NAGL ``openff-gnn-am1bcc-0.1.0-rc.3.pt``; ``"openeye"``:
        OpenEye AM1BCC-ELF10; ``"default_ff_charges"``: Charges as specified in
        force field.
    n_processes
        Number of parallel processes to use to parametrize systems. If ``None``
        or not specified, use all available cores.

    Returns
    -------
    smiles_to_interchange_map
        Dictionary mapping from mapped SMILES strings to the corresponding
        ``Interchange``.

    """
    logger.info(f"Loading dataset at {path}...")
    dataset = datasets.Dataset.load_from_disk(str(path))
    all_smiles: list[str] = dataset["smiles"]

    logger.info("Loaded.")

    logger.info("Constructing interchanges...")

    # get_context("fork") causes memory spike on subsequent calls as memory of
    # previous calls is duplicated. get_context("forkserver") avoids this, but
    # breaks when file is called __main__.py
    with multiprocessing.get_context("forkserver").Pool(
        processes=n_processes,
        initializer=set_logger,
        initargs=(logger,),
    ) as pool:
        maybe_interchanges = list(
            pool.imap(
                functools.partial(
                    smiles_to_interchange,
                    force_field=force_field,
                    charge_model=charge_model,
                ),
                tqdm(all_smiles, desc="Parametrizing"),
                chunksize=1,
            ),
        )
    logger.info("Interchanges constructed.")

    return {
        smiles: interchange
        for smiles, interchange in zip(all_smiles, maybe_interchanges)
        if interchange is not None
    }


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
) -> Interchange | None:
    """
    Parametrize the molecule described by a SMILES string with SMIRNOFF.

    Supports specifying the charge model to use. If the ``"nagl"`` charge model
    is used, and an exception is raised during charge assignment, a warning
    is issued and ``None`` is returned as filtering out `None` objects is the
    easiest way to filter forbidden NAGL chemistries.

    Parameters
    ----------
    smiles
        A mapped SMILES string representing the molecule to parametrize.
    force_field
        A SMIRNOFF Force Field object used to parametrize the molecule.
    charge_model
        Charge model to parametrize with. ``"ambertools"``: AmberTools AM1BCC;
        ``"nagl"``: NAGL ``openff-gnn-am1bcc-0.1.0-rc.3.pt``; ``"openeye"``:
        OpenEye AM1BCC-ELF10; ``"default_ff_charges"``: Charges as specified in
        force field.

    Returns
    -------
    maybe_interchange
        The ``Interchange`` corresponding to the given SMILES parametrized with
        the given force field and charge model, or ``None`` if NAGL raised a
        ``ValueError`` during charge assignment.
    """
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

            try:
                molecule.assign_partial_charges(
                    partial_charge_method="openff-gnn-am1bcc-0.1.0-rc.3.pt",
                    toolkit_registry=NAGLToolkitWrapper(),
                )
            except ValueError as e:
                logger.warning(
                    f"{smiles} encountered ValueError during parametrization, skipping: {e}",
                )
                return None
            charge_from_molecules = [molecule]

    # Construct interchange
    return Interchange.from_smirnoff(
        force_field,
        [molecule],
        charge_from_molecules=charge_from_molecules,
    )


def interchanges_to_smee(
    smiles_to_interchange_map: Mapping[str, Interchange],
) -> tuple[TensorForceField, dict[str, TensorTopology]]:
    """
    Convert ``Interchange`` objects to a Smee tensor objects.
    """
    logger.info("Converting to tensor format...")
    tensor_force_field, tensor_topologies = smee.converters.convert_interchange(
        list(smiles_to_interchange_map.values()),
    )
    logger.info("Converted.")

    tensor_topology_dict = {
        smiles: tensor_topology
        for smiles, tensor_topology in zip(
            smiles_to_interchange_map.keys(),
            tensor_topologies,
            strict=True,
        )
    }

    return tensor_force_field, tensor_topology_dict


def write_tensor_ff_to_disk(
    output_tensor_ff_path: Path,
    shared_tensor_ff: TensorForceField,
):
    """Write the ``TensorForceField`` to the ``Path`` with ``torch.save``."""
    output_tensor_ff_path.parent.mkdir(exist_ok=True, parents=True)
    torch.save(shared_tensor_ff, output_tensor_ff_path)


def write_tensor_tops_to_disk(
    output_tensor_tops_path: Path,
    tensor_topologies: Mapping[str, TensorTopology],
):
    """Write the ``tensor_topologies`` to the ``Path`` with ``torch.save``."""
    output_tensor_tops_path.parent.mkdir(exist_ok=True, parents=True)

    # Save the whole dictionary
    torch.save(
        tensor_topologies,
        output_tensor_tops_path,
    )


def set_logger(logger_: Logger):
    """
    Helper function to sync logger across process pool with forkserver.

    Example
    -------
    with multiprocessing.get_context("forkserver").Pool(
        initializer=worker.set_logger,
        initargs=(logger, )
    ) as pool:
        pool.imap(...)
    """
    global logger
    logger = logger_


if __name__ == "__main__":
    logger.add(Path(__file__).with_suffix(".py.log"), delay=True, enqueue=True)

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
    with logger.catch(onerror=lambda _: sys.exit(1)):
        app()
