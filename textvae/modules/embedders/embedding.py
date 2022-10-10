from typing import cast

import torch

from textvae.data.vocabulary import PAD_TOKEN, Vocabulary
from textvae.modules.embedders.embedder import Embedder


@Embedder.register("embedding")
class Embedding(Embedder):
    def __init__(self, vocab: Vocabulary, embedding_dim: int) -> None:
        super().__init__()
        vocab_size = len(vocab)
        padding_value = vocab[PAD_TOKEN]
        self._embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_value)

    def get_output_dim(self) -> int:
        return self._embedding.embedding_dim

    def forward(self, inputs: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        return cast(torch.Tensor, self._embedding(inputs))
