import pickle
from os import PathLike
from pathlib import Path
from typing import Optional, Union

import torch
from alive_progress import alive_bar
from colt import Lazy

from textvae.data.archive import Archive
from textvae.data.datamodule import TextVaeDataModule
from textvae.data.sampler import BatchSampler
from textvae.models.textvae import TextVAE


class Trainer:
    def __init__(
        self,
        dataset_filename: Union[str, PathLike],
        datamodule: TextVaeDataModule,
        batch_sampler: BatchSampler,
        model: Lazy[TextVAE],
        optimizer: Optional[Lazy[torch.optim.Optimizer]] = None,
        max_epochs: int = 10,
        device: str = "cpu",
    ) -> None:
        self._dataset_filename = dataset_filename
        self._datamodule = datamodule
        self._model_constructor = model
        self._batch_sampler = batch_sampler
        self._optimizer_constructor = optimizer or Lazy({"type": "torch.optim.Adam", "lr": 1e-3})
        self._max_epochs = max_epochs
        self._device = torch.device(device)

    def train(self, workdir: Union[str, PathLike]) -> Archive:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        dataset = self._datamodule.build_dataset(self._dataset_filename, update_vocab=True)
        dataloader = self._datamodule.build_dataloader(dataset, self._batch_sampler)

        model = self._model_constructor.construct(vocab=self._datamodule.vocab)
        optimizer = self._optimizer_constructor.construct(params=model.parameters())

        training_state = Archive(
            datamodule=self._datamodule,
            model=model,
            optimizer=optimizer,
        )

        model.to(device=self._device)
        model.train()

        try:
            with alive_bar(self._max_epochs * len(dataloader), title="Training") as bar:
                for epoch in range(self._max_epochs):
                    total_loss = 0.0
                    num_samples = 0
                    for batch in dataloader:
                        optimizer.zero_grad()
                        batch = batch.to(device=self._device)
                        output = model(batch)
                        loss = output["loss"]
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * len(batch)
                        num_samples += len(batch)
                        bar.text = f" -> Epoch: {epoch}/{self._max_epochs} Loss: {total_loss / num_samples if num_samples else 0:.4f}"
                        bar()
        finally:
            with open(workdir / "archive.pkl", "wb") as pklfile:
                pickle.dump(training_state, pklfile)

        return training_state
