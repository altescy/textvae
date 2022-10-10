import math
import random
from typing import Generic, Iterator, List, Sequence, Sized, TypeVar

import colt

T = TypeVar("T")
T_sized = TypeVar("T_sized", bound=Sized)


class BatchSampler(Generic[T], colt.Registrable):
    def __call__(self, instances: Sequence[T]) -> Iterator[List[T]]:
        raise NotImplementedError

    def get_num_batches(self, instances: Sequence[T]) -> int:
        raise NotImplementedError


@BatchSampler.register("simple")
class SimpleBatchSampler(BatchSampler[T]):
    def __init__(
        self,
        batch_size: int,
        shuffle: bool = False,
    ) -> None:
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __call__(self, instances: Sequence[T]) -> Iterator[List[T]]:
        indices = list(range(len(instances)))

        if self._shuffle:
            random.shuffle(indices)

        for i in range(0, len(instances), self._batch_size):
            yield [instances[j] for j in indices[i : i + self._batch_size]]

    def get_num_batches(self, instances: Sequence[T]) -> int:
        return math.ceil(len(instances) / self._batch_size)


@BatchSampler.register("bucket")
class BucketBatchSampler(BatchSampler[T_sized]):
    def __init__(
        self,
        batch_size: int,
        shuffle: bool = False,
    ) -> None:
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __call__(self, instances: Sequence[T_sized]) -> Iterator[List[T_sized]]:
        sorted_indices = sorted(range(len(instances)), key=lambda i: len(instances[i]), reverse=True)
        buckets: List[List[int]] = []
        bucket: List[int] = []

        for index in sorted_indices:
            if len(bucket) == self._batch_size:
                buckets.append(bucket)
                bucket = []
            bucket.append(index)
        if bucket:
            buckets.append(bucket)

        if self._shuffle:
            random.shuffle(buckets)

        for bucket in buckets:
            yield [instances[i] for i in bucket]

    def get_num_batches(self, instances: Sequence[T_sized]) -> int:
        return math.ceil(len(instances) / self._batch_size)
