from os import PathLike
from typing import Iterator, List, NamedTuple, Optional, Union

import torch
import torchtext.transforms as T
from torch.utils.data import DataLoader, Dataset

from textvae.data.tokenizers import Tokenizer, WhitespaceTokenizer
from textvae.data.vocabulary import EOS_TOKEN, PAD_TOKEN, Vocabulary

Item = List[str]


class Batch(NamedTuple):
    tokens: torch.LongTensor
    mask: torch.BoolTensor

    def to(self, device: torch.device) -> "Batch":
        self.tokens.to(device=device)
        return self

    def __len__(self) -> int:
        return len(self.tokens)


class TextVaeDataset(Dataset[Item]):
    def __init__(self, items: List[Item]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Item:
        return self.items[index]

    def __iter__(self) -> Iterator[Item]:
        return iter(self.items)


class TextVaeDataModule:
    def __init__(
        self,
        vocab: Optional[Vocabulary] = None,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        self._vocab = vocab or Vocabulary()
        self._tokenizer = tokenizer or WhitespaceTokenizer()

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    def build_dataset(
        self,
        filename: Union[str, PathLike],
        *,
        update_vocab: bool = False,
    ) -> TextVaeDataset:
        items: List[List[str]] = []

        def yield_tokens() -> Iterator[List[str]]:
            with open(filename, "r") as f:
                for line in f:
                    item = self._tokenizer.tokenize(line) + [EOS_TOKEN]
                    items.append(item)
                    yield item

        if update_vocab:
            self._vocab.build(yield_tokens())
        else:
            list(yield_tokens())

        return TextVaeDataset(items)

    def build_dataloader(
        self,
        dataset: TextVaeDataset,
        *,
        batch_size: int = 32,
        shuffle: bool = False,
    ) -> DataLoader:
        token_transforms = T.Sequential(
            T.VocabTransform(self.vocab.vocab),
            T.ToTensor(padding_value=self.vocab[PAD_TOKEN]),
        )
        mask_transforms = T.ToTensor(padding_value=0, dtype=torch.bool)

        def collate_fn(items: List[Item]) -> Batch:
            tokens = token_transforms(items)
            mask = mask_transforms([[1] * len(tokens) for tokens in items])
            return Batch(tokens=tokens, mask=mask)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
