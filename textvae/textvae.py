from typing import Dict, List, Optional, Tuple, cast

import torch
from colt import Lazy

from textvae.data.datamodule import Batch
from textvae.data.vocabulary import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, Vocabulary
from textvae.modules.decoders import Decoder
from textvae.modules.embedders import Embedder
from textvae.modules.encoders import Encoder


class TextVAE(torch.nn.Module):
    def __init__(
        self,
        vocab: Vocabulary,
        embedder: Lazy[Embedder],
        encoder: Encoder,
        decoder: Decoder,
        scheduled_sampling_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self._embedder = embedder.construct(vocab=vocab)
        self._encoder = encoder
        self._decoder = decoder
        self._latent_to_mean = torch.nn.Linear(encoder.get_output_dim(), encoder.get_output_dim())
        self._latent_to_logvar = torch.nn.Linear(encoder.get_output_dim(), encoder.get_output_dim())
        self._output_projection = torch.nn.Linear(decoder.get_output_dim(), len(vocab))
        self._bos_index = vocab[BOS_TOKEN]
        self._eos_index = vocab[EOS_TOKEN]
        self._pad_index = vocab[PAD_TOKEN]

        self._scheduled_sampling_ratio = scheduled_sampling_ratio

    @staticmethod
    def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mean

    def encode(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = self._embedder(batch.tokens, batch.mask)
        encodings = self._encoder(embeddings, batch.mask)
        mean = self._latent_to_mean(encodings)
        logvar = self._latent_to_logvar(encodings)
        latent = self.reparameterize(mean, logvar)
        return mean, logvar, latent

    def decode(
        self,
        latent: torch.Tensor,
        *,
        batch: Optional[Batch] = None,
        max_decoding_steps: int = 100,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _ = latent.size()

        inputs = torch.full((batch_size, 1), self._bos_index, dtype=torch.long, device=latent.device)
        mask = cast(torch.BoolTensor, torch.ones((batch_size, 1), dtype=torch.bool, device=latent.device))
        decoder_state = self._decoder.init_state(latent)

        max_decoding_steps = max_decoding_steps if batch is None else batch.tokens.size(1)

        step_logits: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []
        for step in range(max_decoding_steps):
            if self.training and batch is not None and torch.rand(1).item() < self._scheduled_sampling_ratio:
                inputs = batch.tokens[:, step].unsqueeze(1)

            embeddings = self._embedder(inputs, mask).squeeze(1)
            decoder_state, decoder_output = self._decoder(embeddings, latent, decoder_state)
            logits = self._output_projection(decoder_output)
            inputs = logits.argmax(dim=-1)
            mask = cast(torch.BoolTensor, inputs != self._pad_index)

            step_logits.append(logits)
            step_predictions.append(inputs)

            inputs = inputs.unsqueeze(1)
            mask = cast(torch.BoolTensor, mask.unsqueeze(1))

            if batch is None and (inputs == self._eos_index).all():
                break

        logits = torch.stack(step_logits, dim=1)
        predictions = torch.stack(step_predictions, dim=1)
        return logits, predictions

    def _reconstruction_loss(
        self,
        logits: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        batch_size, _, vocab_size = logits.size()
        logits = logits.masked_select(batch.mask.unsqueeze(2)).view(-1, vocab_size)
        targets = batch.tokens.masked_select(batch.mask).view(-1)
        loss = (
            torch.nn.functional.cross_entropy(logits, targets, reduction="sum", ignore_index=self._pad_index)
            / batch_size
        )
        return loss

    def _kdl_loss(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return torch.mean(-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1))

    def loss(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        logits: torch.Tensor,
        batch: Batch,
    ) -> Dict[str, torch.Tensor]:
        reconstruction_loss = self._reconstruction_loss(logits, batch)
        kld_loss = self._kdl_loss(mean, logvar)
        return {"reconstruction_loss": reconstruction_loss, "kld_loss": kld_loss}

    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        mean, logvar, latent = self.encode(batch)
        logits, predictions = self.decode(latent, batch=batch)

        output_dict: Dict[str, torch.Tensor] = {}
        output_dict["mean"] = mean
        output_dict["logver"] = logvar
        output_dict["latent"] = latent
        output_dict["logits"] = logits
        output_dict["predictions"] = predictions

        if self.training:
            loss = self.loss(mean, logvar, logits, batch)
            output_dict["loss"] = loss["reconstruction_loss"] + loss["kld_loss"]

        return output_dict

    def indices_to_tokens(self, vocab: Vocabulary, indices: torch.Tensor) -> List[List[str]]:
        output: List[List[str]] = []
        tensor = indices.detach().cpu()
        for token_indices in tensor:
            tokens: List[str] = []
            for token_index in token_indices:
                if token_index == self._eos_index:
                    break
                tokens.append(vocab.lookup_token(token_index))
            output.append(tokens)
        return output
