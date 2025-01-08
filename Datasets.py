from __future__ import annotations
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len


# Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable
# easy access to the samples.
# The Dataset retrieves our dataset’s features and labels one sample at a time. While training a model, we typically want to
# pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting, and use Python’s multiprocessing
# to speed up data retrieval. DataLoader is an iterable that abstracts this complexity for us in an easy API.

# Dataset =
"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`. Subclasses could also
    optionally implement :meth:`__getitems__`, for speedup batched samples
    loading. This method accepts list of indices of samples of batch and returns
    list of samples.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs an index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
"""

# DataLoader =
"""
Data loader combines a dataset and a sampler, and provides an iterable over the given dataset.
"""
