import argparse
from pathlib import Path

import torch

from textvae.commands.subcommand import Subcommand
from textvae.data.archive import Archive
from textvae.data.sampler import SimpleBatchSampler


@Subcommand.register("reconstruct")
class ReconstructCommand(Subcommand):
    def setup(self) -> None:
        self.parser.add_argument("archive", type=Path)
        self.parser.add_argument("-i", "--input-filename", type=Path)
        self.parser.add_argument("-o", "--output-filename", type=Path)
        self.parser.add_argument("--max-steps", type=int, default=100)
        self.parser.add_argument("--batch-size", type=int, default=32)
        self.parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))

    def run(self, args: argparse.Namespace) -> None:
        archive = Archive.load(args.archive)

        datamodule = archive.datamodule
        dataset = datamodule.build_dataset(args.input_filename)
        dataloader = datamodule.build_dataloader(dataset, SimpleBatchSampler(batch_size=args.batch_size))

        model = archive.model.to(device=args.device).eval()

        with torch.no_grad(), open(args.output_filename, "w") as txtfile:  # type: ignore[no-untyped-call]
            for batch in dataloader:
                batch = batch.to(device=args.device)
                _, _, latent = model.encode(batch)
                _, predictions = model.decode(latent, max_decoding_steps=args.max_steps)
                texts = model.indices_to_tokens(datamodule.vocab, predictions)
                for tokens in texts:
                    print(" ".join(tokens), file=txtfile)
