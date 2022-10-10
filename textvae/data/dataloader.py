from typing import Callable, Generic, Iterator, List, Sequence, TypeVar

from textvae.data.sampler import BatchSampler

T = TypeVar("T")
S = TypeVar("S")


class DataLoader(Generic[T, S]):
    def __init__(
        self,
        instances: Sequence[T],
        batch_sampler: BatchSampler[T],
        collator: Callable[[List[T]], S],
    ) -> None:
        self._instances = instances
        self._batch_sampler = batch_sampler
        self._collator = collator

    def __len__(self) -> int:
        return self._batch_sampler.get_num_batches(self._instances)

    def __iter__(self) -> Iterator[S]:
        for batch in self._batch_sampler(self._instances):
            yield self._collator(batch)
