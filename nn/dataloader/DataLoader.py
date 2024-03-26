import torch
import numpy as np
from ..tensor.Tensor import Tensor
from ..sampler.BatchSampler import BatchSampler


class DataLoader:
    def __init__(self, dataset, sampler=BatchSampler, shuffle=False, batch_size=1, drop_last=False):
        self.dataset = dataset
        self.sampler = sampler(dataset, shuffle, batch_size, drop_last)
 
    def __len__(self):
        return len(self.sampler)
 
    def __call__(self):
        self.__iter__()
 
    def __iter__(self):
        for sample_indices in self.sampler:
            data_list = []
            label_list = []
            for indice in sample_indices:
                data, label = self.dataset[indice]
                data_list.append(data)
                label_list.append(label)
            yield Tensor(np.stack(data_list, axis=0)), Tensor(np.stack(label_list, axis=0))
