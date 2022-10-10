import dataclasses
import pickle
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from alive_progress import alive_bar
from colt import Lazy

from textvae.data.datamodule import TextVaeDataModule
from textvae.textvae import TextVAE


@dataclasses.dataclass
class TrainingState:
    datamodule: TextVaeDataModule
    model: TextVAE
    optimizer: torch.optim.Optimizer


class Trainer:
    def __init__(
        self,
        dataset_filename: Union[str, PathLike],
        datamodule: TextVaeDataModule,
        model: Lazy[TextVAE],
        dataloader: Optional[Dict[str, Any]] = None,
        optimizer: Optional[Lazy[torch.optim.Optimizer]] = None,
        max_epochs: int = 10,
        device: str = "cpu",
    ) -> None:
        self._dataset_filename = dataset_filename
        self._datamodule = datamodule
        self._model_constructor = model
        self._dataloader_config = dataloader or {}
        self._optimizer_constructor = optimizer or Lazy({"type": "torch.optim.Adam", "lr": 1e-3})
        self._max_epochs = max_epochs
        self._device = torch.device(device)

    def train(self, workdir: Union[str, PathLike]) -> TrainingState:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        dataset = self._datamodule.build_dataset(self._dataset_filename, update_vocab=True)
        dataloader = self._datamodule.build_dataloader(dataset, **self._dataloader_config)

        model = self._model_constructor.construct(vocab=self._datamodule.vocab)
        optimizer = self._optimizer_constructor.construct(params=model.parameters())

        training_state = TrainingState(
            datamodule=self._datamodule,
            model=model,
            optimizer=optimizer,
        )

        model.to(device=self._device)
        model.train()

        try:
            for epoch in range(self._max_epochs):
                total_loss = 0.0
                num_samples = 0
                with alive_bar(len(dataloader), title=f"Epoch {epoch}/{self._max_epochs}") as bar:
                    bar.text = f" -> Loss: {total_loss / num_samples if num_samples else 0:.4f}"
                    for batch in dataloader:
                        batch.to(device=self._device)
                        optimizer.zero_grad()
                        output = model(batch)
                        loss = output["loss"]
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item() * len(batch)
                        num_samples += len(batch)
                        bar()
        finally:
            model.eval()
            with open(workdir / "archive.pkl", "wb") as pklfile:
                pickle.dump(training_state, pklfile)

        return training_state
