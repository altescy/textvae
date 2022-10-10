import argparse
import sys
from pathlib import Path
from typing import Iterator

import torch

from textvae.commands.subcommand import Subcommand
from textvae.textvae import TextVAE


@Subcommand.register("reconstruct")
class ReconstructCommand(Subcommand):
    def setup(self) -> None:
        self.parser.add_argument("archive", type=Path)
        self.parser.add_argument("-i", "--input-filename", type=Path, default=None)
        self.parser.add_argument("-o", "--output-filename", type=Path, default=None)
        self.parser.add_argument("-t", "--input-text", action="append", default=[])
        self.parser.add_argument("--max-steps", type=int, default=100)
        self.parser.add_argument("--batch-size", type=int, default=32)
        self.parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))

    def load_texts(self, args: argparse.Namespace) -> Iterator[str]:
        if args.input_filename is not None:
            with args.input_filename.open() as infile:
                yield from (line.strip() for line in infile)
        yield from args.input_text

    def run(self, args: argparse.Namespace) -> None:
        textvae = TextVAE.from_archive(args.archive)
        textvae.set_device(args.device)

        txtfile = open(args.output_filename, "w") if args.output_filename else sys.stdout
        with txtfile:
            texts = self.load_texts(args)
            reconstructions = textvae.reconstruct(
                texts,
                max_decoding_steps=args.max_steps,
                batch_size=args.batch_size,
            )
            for tokens in reconstructions:
                print(" ".join(tokens), file=txtfile)
