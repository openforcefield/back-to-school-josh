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

from loguru import logger


def main(*, flag: bool = True):
    """
    TODO: write me!

    Parameters
    ----------
    flag
        Help text for flag
    """

    pass


if __name__ == "__main__":
    import cyclopts

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
