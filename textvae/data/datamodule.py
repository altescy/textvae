from os import PathLike
from typing import Iterable, Iterator, List, NamedTuple, Optional, Union

import torch
import torchtext.transforms as T

from textvae.data.dataloader import DataLoader
from textvae.data.sampler import BatchSampler
from textvae.data.tokenizers import Tokenizer, WhitespaceTokenizer
from textvae.data.vocabulary import EOS_TOKEN, PAD_TOKEN, Vocabulary

Item = List[str]
TextVaeDataset = List[Item]


class Batch(NamedTuple):
    tokens: torch.LongTensor
    mask: torch.BoolTensor

    def to(self, device: torch.device) -> "Batch":
        self.tokens.to(device=device)
        return self

    def __len__(self) -> int:
        return len(self.tokens)


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

    def build_dataset_from_texts(
        self,
        texts: Iterable[str],
        *,
        update_vocab: bool = False,
    ) -> TextVaeDataset:
        dataset: TextVaeDataset = []

        def yield_tokens() -> Iterator[List[str]]:
            for text in texts:
                item = self._tokenizer.tokenize(text) + [EOS_TOKEN]
                dataset.append(item)
                yield item

        if update_vocab:
            self._vocab.build(yield_tokens())
        else:
            list(yield_tokens())

        return dataset

    def build_dataset(
        self,
        filename: Union[str, PathLike],
        *,
        update_vocab: bool = False,
    ) -> TextVaeDataset:
        def read_file() -> Iterator[str]:
            with open(filename, "r") as txtfile:
                yield from txtfile

        return self.build_dataset_from_texts(
            texts=read_file(),
            update_vocab=update_vocab,
        )

    def build_dataloader(
        self,
        dataset: TextVaeDataset,
        batch_sampler: BatchSampler[Item],
    ) -> DataLoader:
        token_transforms = T.Sequential(
            T.VocabTransform(self.vocab.vocab),
            T.ToTensor(padding_value=self.vocab[PAD_TOKEN]),
        )
        mask_transforms = T.ToTensor(padding_value=0, dtype=torch.bool)

        def collator(items: List[Item]) -> Batch:
            tokens = token_transforms(items)
            mask = mask_transforms([[1] * len(tokens) for tokens in items])
            return Batch(tokens=tokens, mask=mask)

        return DataLoader(
            dataset,
            collator=collator,
            batch_sampler=batch_sampler,
        )
