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


def main():
    pass


if __name__ == "__main__":
    logger.add(Path(__file__).with_suffix(".py.log"), delay=True)
    app = typer.Typer(
        add_completion=False,
        rich_markup_mode="rich",
        pretty_exceptions_enable=False,
    )
    app.command(help=__doc__)(main)
    app()
