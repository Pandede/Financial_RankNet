import abc
import random
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ['ReturnDataset', 'VolatilityDataset', 'CorrelationDataset']


class Delegate:
    def __init__(self, src: Union[Dict[str, str], List, Tuple], formation: int, holding: int, delay: int = 1):
        if isinstance(src, (list, tuple)):
            self.name, self.path = list(range(len(src))), src
        else:
            self.name, self.path = zip(*src.items())
        self.data = torch.stack([torch.from_numpy(np.load(p)) for p in self.path]).float()
        self.formation = formation
        self.holding = holding
        self.delay = max(delay, 1)

        self.c, self.s, self.t = self.data.size()

    def __len__(self):
        return self.t - self.formation - self.holding - self.delay

    def get(self, index: int) -> Tuple[torch.Tensor]:
        formation_data = self.data[..., index:index + self.formation]
        holding_data = self.data[..., index + self.formation + self.delay:index + self.formation + self.holding + self.delay]

        return formation_data, holding_data


class ParentDataset(Dataset, Delegate, abc.ABC):
    def __init__(self, dataset_idx: int, *args, **kwargs):
        Dataset.__init__(self)
        Delegate.__init__(self, *args, **kwargs)

        self.dataset_idx = dataset_idx

    @abc.abstractmethod
    def reduce(self, data):
        return NotImplemented

    @abc.abstractmethod
    def map(self, data_i, data_j):
        return NotImplemented

    def reducedmap(self, data_i: torch.Tensor, data_j: torch.Tensor) -> torch.LongTensor:
        return self.map(self.reduce(data_i), self.reduce(data_j))

    def __getitem__(self, index):
        formation_data, holding_data = self.get(index)
        i, j = random.sample(range(self.s), 2)

        return formation_data[:, i], formation_data[:, j], self.dataset_idx, self.reducedmap(holding_data[:, i], holding_data[:, j])


class ReturnDataset(ParentDataset):
    def __init__(self, *args, **kwargs):
        super(ReturnDataset, self).__init__(*args, **kwargs)

    def reduce(self, data):
        return torch.prod(1 + data[0] / 100.)

    def map(self, data_i, data_j):
        return torch.FloatTensor([data_i > data_j])


class VolatilityDataset(ParentDataset):
    def __init__(self, *args, **kwargs):
        super(VolatilityDataset, self).__init__(*args, **kwargs)

    def reduce(self, data):
        return torch.var(1 + data[0] / 100.)

    def map(self, data_i, data_j):
        return torch.FloatTensor([data_i > data_j])


class CorrelationDataset(ParentDataset):
    def __init__(self, *args, **kwargs):
        super(CorrelationDataset, self).__init__(*args, **kwargs)

    def reduce(self, data):
        return 1 + data[0] / 100.

    def map(self, data_i, data_j):
        data_i[0] += 1e-6
        data_j[0] += 1e-6
        corr = np.corrcoef(data_i, data_j)[0, 1]
        return torch.FloatTensor([corr > 0])
