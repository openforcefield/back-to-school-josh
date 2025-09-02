#!/usr/bin/env python3
"""
TODO: Write me!

Output files:
    file1
        Description of file1
    file2
        Description of file2
"""

from pathlib import Path

import typer
from loguru import logger
from openff.qcsubmit.results.results import (
    BasicResultCollection,
    OptimizationResultCollection,
)
from openff.toolkit import ForceField


def main():
    collection = select_hessians()

    records = collection.to_records()

    ff_template = get_parameter_smirks()

    force_field = compute_aam_parameters(records, ff_template)

    write_smirnoff(force_field, Path(__file__).parent / "specific_initial_ff.offxml")


def select_hessians() -> BasicResultCollection | OptimizationResultCollection:
    """
    Select a dataset of hessians from QCArchive.
    """
    raise NotImplementedError()


def get_parameter_smirks():
    """
    Get a force field that has the SMIRKS parameters we want to fill in with AAM
    """
    raise NotImplementedError()


def compute_aam_parameters(records, ff_template) -> ForceField:
    """
    Compute initial valence parameters with the Alice Allen Method.

    The Alice Allen Method is also known as the Modified Seminario Method (MSM),
    but is here called AAM because calling a technique that has become
    synonymous with initial parameter generation after a non-author of that
    technique that gives the technique an initialism with another much more
    widely used interpretation (Markov State Models) is frankly ridiculous.

    """
    raise NotImplementedError()


def write_smirnoff(ff: ForceField, filename: str | Path):
    """
    Write force field to disk.
    """
    raise NotImplementedError()


if __name__ == "__main__":
    logger.add(Path(__file__).with_suffix(".py.log"), delay=True)
    app = typer.Typer(
        add_completion=False,
        rich_markup_mode="rich",
        pretty_exceptions_enable=False,
    )
    app.command(help=__doc__)(main)
    app()
