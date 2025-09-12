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

from pathlib import Path
from pprint import pformat
from typing import Any, Literal
from collections.abc import Sequence

import datasets
import descent.targets.energy
import descent.train
import json5
import more_itertools
import tensorboardX
import torch
from datasets import Dataset
from loguru import logger
from smee import TensorForceField, TensorTopology
from tqdm import tqdm


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

    train_force_field(
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
        batch_size=batch_size,
        device="cuda" if device == "gpu" else device,
    )


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
    n_epochs: int = 1000,
    learning_rate: float = 0.001,
    batch_size: int = 500,
    device: Literal["cpu", "cuda"] | None = None,
) -> None:
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
    - Creates my-smee-fit/ directory with TensorBoard logs
    - Saves force field checkpoints every 10 epochs as .pt files
    - Saves final optimized force field as final-force-field.pt
    - Logs training metrics (loss, RMSE) to TensorBoard

    Loss function: L = mean(E_pred - E_ref)² + mean(F_pred - F_ref)²
    where energies and forces are weighted equally.
    """
    directory = Path("my-smee-fit")
    directory.mkdir(exist_ok=True, parents=True)

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

    # Load all topologies onto the GPU
    # We might be able to fit larger batches into VRAM if we did this per-batch,
    # but it would be a bit slower?
    topologies = {k: v.to(device) for k, v in topologies.items()}

    trainable = descent.train.Trainable(
        force_field=tensor_force_field.to(device),
        parameters=parameters,
        attributes=attributes,
    )

    trainable_parameters = trainable.to_values().to(device)

    logger.info("Start training...")
    with tensorboardX.SummaryWriter(str(directory)) as writer:
        optimizer = torch.optim.Adam(
            [trainable_parameters],
            lr=learning_rate,
            amsgrad=True,
        )
        dataset_indices = list(range(len(train_data)))

        logger.info("Beginning first epoch")
        for i in tqdm(range(n_epochs), desc="Running epochs"):
            ff = trainable.to_force_field(trainable_parameters)
            epoch_loss = torch.zeros(size=(1,), device=device)
            energy_loss = torch.zeros(size=(1,), device=device)
            force_loss = torch.zeros(size=(1,), device=device)
            grad = None

            for batch_ids in more_itertools.batched(
                tqdm(dataset_indices, leave=False, desc="systems"),
                batch_size,
            ):
                batch = train_data.select(indices=batch_ids).with_format(
                    "torch",
                    device=device,
                )
                true_batch_size = len(batch)
                e_ref, e_pred, f_ref, f_pred = descent.targets.energy.predict(
                    batch,
                    ff,
                    topologies,
                    "mean",
                )
                # L2 loss
                batch_loss_energy = ((e_pred - e_ref) ** 2).sum() / true_batch_size
                batch_loss_force = ((f_pred - f_ref) ** 2).sum() / true_batch_size

                # Equal sum of L2 loss on energies and forces
                batch_loss = batch_loss_energy + batch_loss_force

                (batch_grad,) = torch.autograd.grad(
                    batch_loss,
                    trainable_parameters,
                    create_graph=True,
                )
                batch_grad = batch_grad.detach()
                if grad is None:
                    grad = batch_grad
                else:
                    grad += batch_grad

                # keep sum of squares to report MSE at the end
                epoch_loss += batch_loss.detach()
                energy_loss += batch_loss_energy.detach()
                force_loss += batch_loss_force.detach()

            trainable_parameters.grad = grad

            tqdm.write(f"current epoch loss is {epoch_loss.detach().item()}")

            writer.add_scalar("loss", epoch_loss.detach().item(), i)
            writer.add_scalar("loss_energy", energy_loss.detach().item(), i)
            writer.add_scalar("loss_forces", force_loss.detach().item(), i)

            writer.add_scalar("rmse_energy", energy_loss.detach().sqrt().item(), i)
            writer.add_scalar("rmse_forces", force_loss.detach().sqrt().item(), i)
            writer.flush()

            optimizer.step()
            optimizer.zero_grad()

            if i % 10 == 0:
                torch.save(
                    trainable.to_force_field(trainable_parameters),
                    directory / f"force-field-epoch-{i}.pt",
                )

        logger.info(f"Saving {directory / 'final-force-field.pt'}")
        torch.save(
            trainable.to_force_field(trainable_parameters),
            directory / "final-force-field.pt",
        )


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
