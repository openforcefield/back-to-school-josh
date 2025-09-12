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

import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from pprint import pformat
from typing import Any, Literal

import datasets
import descent.targets.energy
import descent.train
import json5
import tensorboardX
import torch
from datasets import Dataset
from loguru import logger
from openff.toolkit import ForceField
from smee import TensorForceField, TensorTopology
from tqdm import tqdm

from back_to_school_josh.utils import sibpath


def main(
    *,
    tensor_ff_path: Path,
    tensor_tops_path: Path,
    train_dataset_paths: Sequence[Path],
    test_dataset_paths: Sequence[Path],
    training_config_json_path: Path,
    n_epochs: int = 1000,
    learning_rate: float = 0.001,
    batch_size: int = 500,
    device: Literal["gpu", "cpu", None] = None,
    vram_limit_fraction: float = 1.0,
    fitting_dir_path: Path = sibpath("my-smee-fit"),
    smirnoff_template: str = "openff-2.2.1.offxml",
):
    """
    TODO: write me!

    Parameters
    ----------
    tensor_ff_path
        Path to Smee tensor force field saved with ``torch.save(...)``.
    tensor_tops_path
        Path to dictionary, saved with ``torch.save(...)``, mapping from mapped
        SMILES strings to Smee tensor topologies indexing into the tensor force
        field at ``tensor_ff_path``.
    train_dataset_paths
        List of paths to Smee Huggingface datasets that will be concatenated to
        form the training set.
    test_dataset_paths
        List of paths to Smee Huggingface datasets that will be concatenated to
        form the test set.
    training_config_json_path
        JSON5 file consisting of a single object with keys ``"parameters"`` and
        ``"attributes"`` that can be assigned to the ``"parameters"`` and
        ``"attributes"`` arguments of ``descent.train.Trainable``.
    n_epochs
        Number of epochs to train for.
    learning_rate
        Learning rate for Adam optimizer.
    batch_size
        Number of molecular configurations per batch.
    device
        Device to use to perform optimization; ``None`` or unspecified uses GPU
        if available, but falls back to CPU.
    vram_limit_fraction
        The fraction of available VRAM this fit should limit itself to. PyTorch
        keeps a cache of values in VRAM that can grow quite quickly, so setting
        a limit of around 50%-80% can be useful on machines that have to do more
        than one things with their GPUs (eg, display a desktop).
    fitting_dir_path
        Path to save checkpoints and tensorboard logs to
    """
    logger.info("---------------------- starting script ----------------------")

    logger.info("Loading tensor force field")
    tensor_ff: TensorForceField = torch.load(tensor_ff_path, weights_only=False)

    logger.info("Loading tensor topologies")
    tensor_tops: dict[str, TensorTopology] = torch.load(
        tensor_tops_path,
        weights_only=False,
    )

    logger.info("Loading training dataset(s)")
    train_dataset = datasets.concatenate_datasets(
        [Dataset.load_from_disk(str(path)) for path in train_dataset_paths],
    )

    logger.info("Checking training dataset(s)")
    train_dataset = remove_missing_rows(train_dataset, tensor_tops)

    logger.info("Loading test dataset(s)")
    test_dataset = datasets.concatenate_datasets(
        [Dataset.load_from_disk(str(path)) for path in test_dataset_paths],
    )

    logger.info("Checking test dataset(s)")
    test_dataset = remove_missing_rows(test_dataset, tensor_tops)

    training_config = json5.loads(training_config_json_path.read_text())

    logger.info(f"Training config: {pformat(training_config)}")

    torch.cuda.set_per_process_memory_fraction(vram_limit_fraction)

    trained_force_field = train_force_field(
        train_data=train_dataset,
        test_data=test_dataset,
        tensor_force_field=tensor_ff,
        parameters={
            k: descent.train.ParameterConfig(**v)
            for k, v in training_config["parameters"].items()  # type: ignore
        },
        attributes={
            k: descent.train.AttributeConfig(**v)
            for k, v in training_config["attributes"].items()  # type: ignore
        },
        topologies=tensor_tops,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device="cuda" if device == "gpu" else device,
        directory=fitting_dir_path,
    )

    logger.info("Converting final FF to SMIRNOFF")
    final_smirnoff = tensor_ff_to_smirnoff(
        trained_force_field,
        ForceField(smirnoff_template),
    )
    final_smirnoff_path = fitting_dir_path / "final.offxml"
    logger.info(f"Saving fitted SMIRNOFF force field to {final_smirnoff_path}")
    final_smirnoff.to_file(str(final_smirnoff_path))


def remove_missing_rows(dataset: Dataset, smiles_to_keep: dict[str, Any]) -> Dataset:
    """Return a dataset with rows missing from smiles_to_keep removed."""
    keep_indices: list[int] = []
    skip_smiles: list[str] = []
    for i, smiles in enumerate(dataset["smiles"]):
        if smiles in smiles_to_keep:
            keep_indices.append(i)
        else:
            skip_smiles.append(smiles)
    if len(skip_smiles) != 0:
        smiles_lines = "".join("\n    " + smiles for smiles in skip_smiles)
        logger.warning(
            f"Skipping some SMILES because their topologies do not exist:{smiles_lines}",
        )

    return dataset.select(keep_indices)


def train_force_field(
    *,
    train_data: Dataset,
    test_data: Dataset,
    tensor_force_field: TensorForceField,
    parameters: dict[str, descent.train.ParameterConfig],
    attributes: dict[str, descent.train.AttributeConfig],
    topologies: dict[str, TensorTopology],
    n_epochs: int,
    learning_rate: float,
    batch_size: int,
    device: Literal["cpu", "cuda"] | None,
    directory: Path,
) -> TensorForceField:
    """Train force field parameters using molecular energy and force data.

    Optimizes force field parameters by minimizing the sum of squared errors
    between predicted and reference energies and forces using gradient descent
    with the
    `Adam optimizer<https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html>`_.

    Parameters
    ----------
    train_filename_data
        Training dataset in Huggingface format.
    test_filename_data
        Test dataset in Huggingface format.
    smee_force_field
        SMEE force field tensor object with parameters to optimize.
    topologies
        Dictionary mapping SMILES strings to SMEE topology tensor objects.
    descent_config
        Configuration object for Descent
    n_epochs
        Number of training epochs.
    learning_rate
        Learning rate for Adam optimizer.
    batch_size
        Number of molecular configurations per batch.

    Notes
    -----
    Side effects:
    - Creates ``directory`` directory with TensorBoard logs
    - Saves force field checkpoints every 10 epochs as .pt files in ``directory``
    - Saves final optimized force field as ``directory / "final-force-field.pt"``
    - Logs training metrics (loss, RMSE) to TensorBoard
    - Writes progress bars and logs to stdout/logger

    Loss function: L = mse(E_pred - E_ref) + mse(F_pred - F_ref)
    where energies and forces are weighted equally and mse(x) is the mean
    squared error of x.
    """
    directory.mkdir(exist_ok=True, parents=True)

    logger.info(
        f"Will fit for {n_epochs} epochs with constant learning rate {learning_rate} and batch size {batch_size}",
    )

    logger.info(f"{torch.cuda.is_available()=}")
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    logger.info(f"Training on {device}")
    if device == "cuda":
        logger.info("NB: AMD GPUs using ROCm also appear as cuda")
        logger.info(
            f"First available cuda/ROCm device: {torch.cuda.get_device_name(0)}",
        )

    logger.info(f"Copying topologies to {device} memory...")
    topologies = {k: v.to(device) for k, v in topologies.items()}
    logger.info("Copied.")

    # Get a tensor of the actual numbers we're going to optimize
    logger.info("Initializing trainable_parameters...")
    trainable = descent.train.Trainable(
        force_field=tensor_force_field.to(device),
        parameters=parameters,
        attributes=attributes,
    )
    trainable_parameters = trainable.to_values().to(device)
    logger.info("Initialized.")

    # Prepare batching
    train_dataloader = torch.utils.data.DataLoader(
        train_data,  # type: ignore
        batch_size=batch_size,
        collate_fn=lambda samples: samples,
        pin_memory=True,
    )

    logger.info("Start training...")
    with tensorboardX.SummaryWriter(str(directory)) as writer:
        optimizer = torch.optim.Adam(
            [trainable_parameters],
            lr=learning_rate,
            amsgrad=True,
        )

        epoch_tqdm = tqdm(range(n_epochs), desc="epochs")
        for i in epoch_tqdm:
            ff = trainable.to_force_field(trainable_parameters)
            epoch_loss = torch.zeros(size=(1,), device=device)
            energy_loss = torch.zeros(size=(1,), device=device)
            force_loss = torch.zeros(size=(1,), device=device)
            grad = None

            batch_tqdm = tqdm(
                leave=False,
                desc="computing loss",
                total=len(train_data),
                unit="tops",
            )
            for cpu_batch in train_dataloader:
                # Copy the batch to GPU
                batch = [
                    {k: v if k == "smiles" else v.to(device) for k, v in sample.items()}
                    for sample in cpu_batch
                ]
                true_batch_size = len(batch)
                # Compute forces and energies
                e_ref, e_pred, f_ref, f_pred = descent.targets.energy.predict(
                    batch,  # type: ignore
                    ff,
                    topologies,
                    "mean",
                )
                # Compute L2 loss
                batch_loss_energy = ((e_pred - e_ref) ** 2).sum() / true_batch_size
                batch_loss_force = ((f_pred - f_ref) ** 2).sum() / true_batch_size

                # Equal sum of L2 loss on energies and forces
                batch_loss = batch_loss_energy + batch_loss_force

                # Compute the gradient of batch_loss wrt trainable_parameters
                (batch_grad,) = torch.autograd.grad(
                    batch_loss,
                    trainable_parameters,
                    create_graph=True,
                )
                # Add the batch gradient to the cumulative epoch gradient
                batch_grad = batch_grad.detach()
                if grad is None:
                    grad = batch_grad
                else:
                    grad += batch_grad

                # keep cumulative epoch losses to report MSE at the end
                epoch_loss += batch_loss.detach()
                energy_loss += batch_loss_energy.detach()
                force_loss += batch_loss_force.detach()

                # Update the progress bar
                batch_tqdm.update(true_batch_size)
            batch_tqdm.close()

            # Write results to logs
            epoch_tqdm.set_description(
                f"loss: {epoch_loss.detach().item()}, epochs",
            )

            writer.add_scalar("loss", epoch_loss.detach().item(), i)
            writer.add_scalar("loss_energy", energy_loss.detach().item(), i)
            writer.add_scalar("loss_forces", force_loss.detach().item(), i)

            writer.add_scalar("rmse_energy", energy_loss.detach().sqrt().item(), i)
            writer.add_scalar("rmse_forces", force_loss.detach().sqrt().item(), i)
            writer.flush()

            # Perform the optimization step
            trainable_parameters.grad = grad
            optimizer.step()
            optimizer.zero_grad()

            if i % 10 == 0:
                torch.save(
                    trainable.to_force_field(trainable_parameters),
                    directory / f"force-field-epoch-{i}.pt",
                )

        logger.info(f"Saving {directory / 'final-force-field.pt'}")
        final_force_field = trainable.to_force_field(trainable_parameters)
        torch.save(
            final_force_field,
            directory / "final-force-field.pt",
        )
    return final_force_field


def tensor_ff_to_smirnoff(
    smee_force_field: TensorForceField,
    offxml_template: ForceField,
) -> ForceField:
    """Convert optimized SMEE force field parameters to SMIRNOFF format.

    Takes the optimized parameters from a SMEE force field and writes them
    back to an OpenFF ``ForceField`` object, preserving the original force field
    structure while updating the fitted parameters.

    Parameters
    ----------
    smee_force_field : smee.TensorForceField
        Optimized SMEE force field tensor object containing fitted parameters.
    offxml : pathlib.Path | str
        Path to the reference OFFXML file used for output structure.

    Returns
    -------
    None

    Notes
    -----
    Parameter handling by type:
    - Bonds/Angles: Updates k (force constant) and equilibrium values
    - ProperTorsions: Collects k values by periodicity for each SMIRKS pattern
    - ImproperTorsions: Updates only the k values (v2 terms)
    """
    new_ff = ForceField(offxml_template.to_string())

    for potential in smee_force_field.potentials:
        handler_name = potential.parameter_keys[0].associated_handler

        parameter_attrs = potential.parameter_cols
        parameter_units = potential.parameter_units

        if handler_name in ["Bonds", "Angles"]:
            handler = new_ff.get_parameter_handler(handler_name)
            for i, opt_parameters in enumerate(potential.parameters):
                smirks = potential.parameter_keys[i].id
                ff_parameter = handler[smirks]
                opt_parameters = opt_parameters.detach().cpu().numpy()
                for j, (p, unit) in enumerate(zip(parameter_attrs, parameter_units)):
                    setattr(ff_parameter, p, opt_parameters[j] * unit)

        elif handler_name in ["ProperTorsions"]:
            handler = new_ff.get_parameter_handler(handler_name)
            k_index = parameter_attrs.index("k")
            p_index = parameter_attrs.index("periodicity")
            # we need to collect the k values into a list across the entries
            collection_data: dict[str, dict[int, float]] = defaultdict(dict)
            for i, opt_parameters in enumerate(potential.parameters):
                smirks = potential.parameter_keys[i].id
                ff_parameter = handler[smirks]
                opt_parameters = opt_parameters.detach().cpu().numpy()
                # find k and the periodicity
                k = opt_parameters[k_index] * parameter_units[k_index]
                p = int(opt_parameters[p_index])
                collection_data[smirks][p] = k
            # now update the force field
            for smirks, k_s in collection_data.items():
                ff_parameter = handler[smirks]
                k_mapped_to_p = [k_s[p] for p in ff_parameter.periodicity]
                ff_parameter.k = k_mapped_to_p

        elif handler_name in ["ImproperTorsions"]:
            k_index = parameter_attrs.index("k")
            handler = new_ff.get_parameter_handler(handler_name)
            # we only fit the v2 terms for improper torsions so convert to list and set
            for i, opt_parameters in enumerate(potential.parameters):
                smirks = potential.parameter_keys[i].id
                ff_parameter = handler[smirks]
                opt_parameters = opt_parameters.detach().cpu().numpy()
                ff_parameter.k = [opt_parameters[k_index] * parameter_units[k_index]]

    return new_ff


if __name__ == "__main__":
    import sys

    import cyclopts

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
