from os import PathLike
from typing import Iterable, Iterator, List, Tuple, Union

import torch

from textvae.data.archive import Archive
from textvae.data.datamodule import TextVaeDataModule
from textvae.data.sampler import SimpleBatchSampler
from textvae.models.textvae import TextVAE as TextVAEModel


class TextVAE:
    def __init__(
        self,
        model: TextVAEModel,
        datamodule: TextVaeDataModule,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        if isinstance(device, str):
            device = torch.device(device)

        self._model = model.to(device=device)
        self._datamodule = datamodule
        self._device = device

    def set_device(self, device: Union[str, torch.device]) -> None:
        if isinstance(device, str):
            device = torch.device(device)
        self._device = device
        self._model.to(device=device)

    def reconstruct(
        self,
        texts: Iterable[str],
        *,
        batch_size: int = 16,
        max_decoding_steps: int = 100,
    ) -> Iterator[List[str]]:
        dataset = self._datamodule.build_dataset_from_texts(texts)
        dataloader = self._datamodule.build_dataloader(dataset, SimpleBatchSampler(batch_size=batch_size))

        self._model.eval()
        self._model.to(device=self._device)
        with torch.no_grad():  # type: ignore[no-untyped-call]
            for batch in dataloader:
                batch = batch.to(device=self._device)
                _, _, latent = self._model.encode(batch)
                _, predictions = self._model.decode(latent, max_decoding_steps=max_decoding_steps)
                reconstructed_texts = self._model.indices_to_tokens(self._datamodule.vocab, predictions)
                yield from reconstructed_texts

    def encode(
        self,
        texts: Iterable[str],
        *,
        batch_size: int = 16,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        dataset = self._datamodule.build_dataset_from_texts(texts)
        dataloader = self._datamodule.build_dataloader(dataset, SimpleBatchSampler(batch_size=batch_size))

        self._model.eval()
        self._model.to(device=self._device)
        with torch.no_grad():  # type: ignore[no-untyped-call]
            for batch in dataloader:
                batch = batch.to(device=self._device)
                mean, logvar, _ = self._model.encode(batch)
                yield from zip(mean, logvar)

    def decode(
        self,
        latents: torch.Tensor,
        *,
        batch_size: int = 16,
        max_decoding_steps: int = 100,
    ) -> Iterator[List[str]]:
        self._model.eval()
        self._model.to(device=self._device)
        with torch.no_grad():  # type: ignore[no-untyped-call]
            for batch in torch.split(latents, batch_size):  # type: ignore[no-untyped-call]
                batch = batch.to(device=self._device)
                _, predictions = self._model.decode(batch, max_decoding_steps=max_decoding_steps)
                decoded_texts = self._model.indices_to_tokens(self._datamodule.vocab, predictions)
                yield from decoded_texts

    @classmethod
    def from_archive(cls, archive: Union[str, PathLike, Archive]) -> "TextVAE":
        if not isinstance(archive, Archive):
            archive = Archive.load(archive)
        return cls(model=archive.model, datamodule=archive.datamodule)
