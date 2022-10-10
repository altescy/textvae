from collections.abc import Sized
from typing import Callable, Generic, Iterator, List, Sequence, TypeVar

T = TypeVar("T")
S = TypeVar("S")


class DataLoader(Generic[T, S]):
    def __init__(
        self,
        instances: Sequence[T],
        batch_sampler: Callable[[Sequence[T]], Iterator[List[T]]],
        collator: Callable[[List[T]], S],
    ) -> None:
        self._instances = instances
        self._batch_sampler = batch_sampler
        self._collator = collator

    def __len__(self) -> int:
        if isinstance(self._instances, Sized):
            return len(self._instances)
        raise TypeError

    def __iter__(self) -> Iterator[S]:
        for batch in self._batch_sampler(self._instances):
            yield self._collator(batch)
