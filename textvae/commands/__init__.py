from __future__ import annotations

import argparse

from textvae import __version__
from textvae.commands import train  # noqa: F401
from textvae.commands.subcommand import Subcommand


def create_subcommand(prog: str | None = None) -> Subcommand:
    parser = argparse.ArgumentParser(usage="%(prog)s", prog=prog)
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s " + __version__,
    )
    return Subcommand(parser)


def main(prog: str | None = None) -> None:
    app = create_subcommand(prog)
    app()
