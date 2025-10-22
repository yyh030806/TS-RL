import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np

class KRepeatSampler(Sampler):
    """
    Sampler that repeats each sample k times and arranges them into batches.

    Args:
        data_set (Dataset): The dataset to sample from.
        repeat_k (int): The number of times to repeat each sample.
        batch_size (int): The size of each batch. The batch size must be divisible by repeat_k.
    """
    def __init__(self, data_set, repeat_k, batch_size):
        
        self.data_set = data_set
        self.repeat_k =  repeat_k
        self.batch_size = batch_size
        
        assert self.batch_size % self.repeat_k == 0, "batch_size must be divisible by repeat_k"

    def __iter__(self):
        """
        Yields batches of indices where each original sample index is repeated k times.
        """
        # 1. Create a list of indices for each sample, repeated k times
        repeated_indices = []
        for i in range(len(self.data_set)):
            repeated_indices.extend([i] * self.repeat_k)
        
        # 2. Shuffle the repeated indices
        np.random.shuffle(repeated_indices)
        
        # 3. Create batches
        for i in range(0, len(repeated_indices), self.batch_size):
            batch = repeated_indices[i:i+self.batch_size]
            if len(batch) == self.batch_size:
                yield batch

    def __len__(self):
        """
        Returns the total number of samples in an epoch.
        """
        return (len(self.data_set) * self.repeat_k) // self.batch_size