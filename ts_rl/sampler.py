import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np

class KRepeatSampler(Sampler):
    """
    Sampler that, for each epoch:
    1. Randomly selects a subset of a specified size from the dataset.
    2. Repeats each index in that subset k times.
    3. Shuffles and yields batches of these repeated indices.

    Args:
        data_set (Dataset): The dataset to sample from.
        max_num (int): The number of unique samples to draw from the dataset for each epoch.
        repeat_k (int): The number of times to repeat each sample from the subset.
        batch_size (int): The size of each batch.
    """
    def __init__(self, data_set: Dataset, max_num: int, repeat_k: int, batch_size: int):
        self.data_set = data_set
        self.max_num = max_num
        self.repeat_k = repeat_k
        self.batch_size = batch_size

        # A good practice assertion to ensure you don't ask for more data than available.
        if self.max_num > len(self.data_set):
            raise ValueError("max_num cannot be larger than the total dataset size.")

        # This assertion is not strictly required but helps create well-formed batches.
        # If your last batch can be smaller, you can remove this.
        if (self.max_num * self.repeat_k) % self.batch_size != 0:
            print(f"Warning: (max_num * repeat_k) is not divisible by batch_size. "
                  f"The last batch might be dropped.")

    def __iter__(self):
        """
        Yields batches of indices for one epoch.
        """
        # 1. Get all possible indices from the dataset.
        all_indices = np.arange(len(self.data_set))

        # 2. Randomly sample a subset of indices WITHOUT replacement. This is the key change.
        subset_indices = np.random.choice(all_indices, self.max_num, replace=False)

        # 3. Create a list where each index from the SUBSET is repeated k times.
        repeated_indices = []
        for i in subset_indices:
            repeated_indices.extend([i] * self.repeat_k)
        
        # 4. Shuffle the final list of repeated indices.
        np.random.shuffle(repeated_indices)
        
        # 5. Create and yield batches.
        for i in range(0, len(repeated_indices), self.batch_size):
            batch = repeated_indices[i:i + self.batch_size]
            # Drop the last batch if it's not a full batch.
            if len(batch) == self.batch_size:
                yield batch

    def __len__(self) -> int:
        """
        Returns the number of batches per epoch.
        """
        # The total number of items processed in an epoch is max_num * repeat_k
        return (self.max_num * self.repeat_k) // self.batch_size