from typing import Iterator, List, Optional, cast

import torch
from torchtext.vocab import Vocab, build_vocab_from_iterator

PAD_TOKEN = "@@PADDING@@"
UNK_TOKEN = "@@UNKNOWN@@"
BOS_TOKEN = "@@BOFSENT@@"
EOS_TOKEN = "@@EOFSENT@@"


class Vocabulary(torch.nn.Module):
    def __init__(
        self,
        min_freq: int = 1,
        max_tokens: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._vocab: Optional[Vocab] = None
        self._min_freq = min_freq
        self._max_tokens = max_tokens

    @property
    def vocab(self) -> Vocab:
        if self._vocab is None:
            raise ValueError("Vocab is not built yet.")
        return self._vocab

    def build(self, iterator: Iterator[List[str]]) -> None:
        self._vocab = build_vocab_from_iterator(
            iterator,
            min_freq=self._min_freq,
            max_tokens=self._max_tokens,
            specials=[PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN],
        )
        self._vocab.set_default_index(self._vocab[UNK_TOKEN])

    def forward(self, tokens: List[str]) -> List[int]:
        return cast(List[int], self.vocab(tokens))

    def __contains__(self, token: str) -> bool:
        return token in self.vocab

    def __getitem__(self, token: str) -> int:
        return cast(int, self.vocab[token])

    def __len__(self) -> int:
        return len(self.vocab)
