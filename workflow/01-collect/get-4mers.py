#!/usr/bin/env python3
"""
Download the 4-mers dataset from QCArchive to disk for later processing.

Output files:
    {data_dir}/optimization-result-collection.json
        JSON serialized result collection
    {data_dir}/records/{record_id}/molecule.sdf
        SDF of molecule
    {data_dir}/records/{record_id}/record.pickle
        Pickled result record
"""

import pickle
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from openff.qcsubmit.results import OptimizationResultCollection
from qcportal import PortalClient
from qcportal.client import SinglepointDriver
from tqdm import tqdm

DEFAULT_DATASETS = [
    "OpenFF Protein PDB 4-mers v4.0",
]


def main(
    data_dir: Annotated[
        Path,
        typer.Argument(help="Directory to download data to"),
    ] = Path(".data/qcarchive"),
    datasets: Annotated[
        list[str] | None,
        typer.Argument(
            help="Datasets to download from QCArchive",
            show_default=" ".join(map(repr, DEFAULT_DATASETS)),
        ),
    ] = None,
):
    data_dir.mkdir(exist_ok=True)

    qc_client = PortalClient(
        "https://api.qcarchive.molssi.org:443",
        cache_dir=".data/.qc_cache",
    )

    opt_result_collection_path = data_dir / "optimization-result-collection.json"

    if opt_result_collection_path.exists():
        optimization_result_collection = OptimizationResultCollection.parse_file(
            opt_result_collection_path,
        )
    else:
        optimization_result_collection = OptimizationResultCollection.from_server(
            client=qc_client,
            datasets=DEFAULT_DATASETS if datasets is None else datasets,
            spec_name="default",
        )
        opt_result_collection_path.write_text(optimization_result_collection.json())

    logger.info(f"N RESULTS:   {optimization_result_collection.n_results}")
    logger.info(f"N MOLECULES: {optimization_result_collection.n_molecules}")

    logger.info("Downloading records")
    # records = download_complete_records(optimization_result_collection)
    records = optimization_result_collection.to_basic_result_collection(
        list(SinglepointDriver),
    ).to_records()
    logger.info("Done, saving")
    for record, molecule in tqdm(records):
        record_dir = data_dir / f"records/{record.id}"
        record_dir.mkdir(exist_ok=True, parents=True)
        molecule.to_file(record_dir / "molecule.sdf", "SDF")
        with open(record_dir / "record.pickle", "wb") as file:
            pickle.dump(record, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    logger.add(Path(__file__).with_suffix(".py.log"), delay=True)
    app = typer.Typer(add_completion=False, rich_markup_mode="rich")
    app.command(help=__doc__)(main)
    app()
