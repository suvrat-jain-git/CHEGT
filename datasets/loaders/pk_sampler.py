"""PK (Person-Sequence) batch sampler for metric learning.

Each batch contains exactly P identities and K sequences per identity,
giving a batch size of P × K.  Triplet loss with hard mining requires
multiple samples per identity within each batch.
"""

import random
from collections import defaultdict
from typing import Dict, Iterator, List

from torch.utils.data import Sampler


class PKSampler(Sampler[int]):
    """Yield P×K indices per batch for *num_iters_per_epoch* batches.

    Args:
        labels:              Flat list of identity labels, one per dataset item.
        P:                   Number of distinct identities per batch.
        K:                   Number of sequences per identity per batch.
        num_iters_per_epoch: Total batches to yield before the epoch ends.
                             Total indices emitted = P × K × num_iters_per_epoch.
    """

    def __init__(
        self,
        labels: List[int],
        P: int,
        K: int,
        num_iters_per_epoch: int = 200,
    ) -> None:
        super().__init__()
        self.P = P
        self.K = K
        self.num_iters = num_iters_per_epoch

        # Build identity → dataset-index mapping
        self.pid_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, pid in enumerate(labels):
            self.pid_to_indices[pid].append(idx)

        self.pids: List[int] = list(self.pid_to_indices.keys())

        if len(self.pids) < P:
            raise ValueError(
                f"PKSampler requires at least P={P} identities, "
                f"but only {len(self.pids)} found."
            )

    # ── Sampler interface ─────────────────────────────────────────────

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.num_iters):
            yield from self._sample_batch()

    def __len__(self) -> int:
        return self.num_iters * self.P * self.K

    # ── Internal ─────────────────────────────────────────────────────

    def _sample_batch(self) -> List[int]:
        chosen_pids = random.sample(self.pids, self.P)
        indices: List[int] = []
        for pid in chosen_pids:
            pool = self.pid_to_indices[pid]
            if len(pool) >= self.K:
                indices.extend(random.sample(pool, self.K))
            else:
                # Sample with replacement when fewer sequences than K
                indices.extend(random.choices(pool, k=self.K))
        return indices
