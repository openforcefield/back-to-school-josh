#!/usr/bin/env python3
""" """

import pickle
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from openff.qcsubmit.results import OptimizationResultCollection
from qcportal import PortalClient
from tqdm import tqdm


def main(
    data_dir: Annotated[
        Path,
        typer.Argument(help="Directory to download data to"),
    ] = Path(".data/qcarchive"),
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
            datasets=[
                "OpenFF Protein PDB 4-mers v4.0",
            ],
            spec_name="default",
        )
        opt_result_collection_path.write_text(optimization_result_collection.json())

    logger.info(f"N RESULTS:   {optimization_result_collection.n_results}")
    logger.info(f"N MOLECULES: {optimization_result_collection.n_molecules}")

    logger.info("Downloading records")
    records = optimization_result_collection.to_records()
    logger.info("Done, saving")
    for record, molecule in tqdm(records):
        record_dir = data_dir / f"records/{record.id}"
        record_dir.mkdir(exist_ok=True, parents=True)
        molecule.to_file(record_dir / "molecule.sdf", "SDF")
        with open(record_dir / "record.pickle", "wb") as file:
            pickle.dump(record, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    app = typer.Typer(add_completion=False)
    app.command(help=__doc__)(main)
    app()
