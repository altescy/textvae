import pickle
from os import PathLike
from typing import NamedTuple, Union

import torch

from textvae.data.datamodule import TextVaeDataModule
from textvae.textvae import TextVAE


class Archive(NamedTuple):
    datamodule: TextVaeDataModule
    model: TextVAE
    optimizer: torch.optim.Optimizer

    def save(self, filename: Union[str, PathLike]) -> None:
        with open(filename, "wb") as pklfile:
            pickle.dump(self, pklfile)

    @classmethod
    def load(cls, filename: Union[str, PathLike]) -> "Archive":
        with open(filename, "rb") as pklfile:
            archive = pickle.load(pklfile)
            assert isinstance(archive, cls)
            return archive
