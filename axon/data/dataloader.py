# Copyright 2025 Model AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torchdata.stateful_dataloader import StatefulDataLoader


class DynamicDataLoader:
    """
    Wraps StatefulDataLoader to support dynamic batch sizes while preserving
    checkpoint/resume functionality.
    
    This loader never exhausts in infinite mode - it automatically wraps around when the dataset
    is exhausted, allowing infinite sampling. Uses internal batching to leverage
    num_workers parallelism.
    """

    def __init__(self, dataset, collate_fn=None, batch_size: int = 1, infinite: bool = True, **kwargs):
        """
        Args:
            dataset: The dataset to load from
            collate_fn: Function to collate samples into batches
            batch_size: Default batch size for iteration
            infinite: If True, loop forever. If False, stop after one pass.
            **kwargs: Additional arguments passed to StatefulDataLoader
        """
        kwargs.pop('batch_sampler', None)
        kwargs.pop('drop_last', None)

        self._dataset = dataset
        self._collate_fn = collate_fn or torch.utils.data.default_collate
        self.batch_size = batch_size
        self.infinite = infinite
        self._exhausted = False

        # Underlying loader fetches batches but defers collation
        # We use _list_collate internally so we can rebatch to any size
        self._loader = StatefulDataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self._list_collate,
            **kwargs
        )
        self._iter = None
        self._buffer = []  # Buffer to hold prefetched samples

    def __iter__(self):
        self._iter = iter(self._loader)
        self._buffer = []
        self._exhausted = False
        return self

    def _refill_buffer(self):
        """Refill buffer from internal loader. Returns True if data was added, False otherwise."""
        if self._exhausted:
            return False
            
        if self._iter is None:
            self._iter = iter(self._loader)
        
        try:
            batch = next(self._iter)
            if batch:
                self._buffer.extend(batch)
                return True
            return False
        except StopIteration:
            if self.infinite:
                self._iter = iter(self._loader)
                try:
                    batch = next(self._iter)
                except StopIteration:
                    self._exhausted = True
                    return False
                self._buffer.extend(batch)
                return True
            else:
                self._exhausted = True
                return False

    def next(self, batch_size: int):
        """
        Fetch next batch with the specified size.
        
        Never raises StopIteration in infinite mode - automatically wraps around for infinite sampling.

        Args:
            batch_size: Number of samples in this batch

        Returns:
            Collated batch of samples
        """
        assert batch_size > 0, f"batch_size has to be positive: {batch_size}"
        
        # Refill buffer until we have enough samples
        while len(self._buffer) < batch_size:
            added = self._refill_buffer()
            if not added:
                break
        
        if len(self._buffer) == 0:
            raise StopIteration()

        # Take requested samples from buffer
        batch = self._buffer[:batch_size]
        self._buffer = self._buffer[batch_size:]

        return self._collate_fn(batch)

    @staticmethod
    def _list_collate(batch):
        """Collate that just returns the list of samples without any processing."""
        return batch

    def __next__(self):
        """Default iteration with batch_size=1."""
        return self.next(batch_size=self.batch_size)

    def __len__(self):
        """Return the length of the underlying dataset."""
        return len(self._dataset)

    @property
    def dataset(self):
        """Access the underlying dataset."""
        return self._dataset

    @property
    def sampler(self):
        """Access the underlying sampler."""
        return self._loader.sampler

    # ---- Stateful API (passthrough to StatefulDataLoader) ----

    def state_dict(self):
        """Get checkpoint state. Includes sampler state if the sampler is stateful."""
        state = {
            'loader_state': self._loader.state_dict(),
            'buffer': self._buffer,
            'exhausted': self._exhausted,
        }
        sampler = self._loader.sampler
        if sampler is not None and hasattr(sampler, "state_dict"):
            state['sampler_state'] = sampler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        """Restore from checkpoint. Restores sampler state if present."""
        self._loader.load_state_dict(state_dict['loader_state'])
        self._buffer = state_dict.get('buffer', [])
        self._exhausted = state_dict.get('exhausted', False)
        self._iter = None  # Will reinitialize on next fetch
        sampler = self._loader.sampler
        if 'sampler_state' in state_dict and sampler is not None and hasattr(sampler, "load_state_dict"):
            sampler.load_state_dict(state_dict['sampler_state'])
