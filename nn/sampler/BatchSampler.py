import numpy as np

class BatchSampler:
    def __init__(self, dataset=None, shuffle=False, batch_size=1, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
 
        self.num_data = len(dataset)
        if self.drop_last or (self.num_data % batch_size == 0):
            self.num_samples = self.num_data // batch_size
        else:
            self.num_samples = self.num_data // batch_size + 1
        indices = np.arange(self.num_data)
        if shuffle:
            np.random.shuffle(indices)
        if drop_last:
            indices = indices[:self.num_samples * batch_size]
        self.indices = indices
 
    def __len__(self):
        return self.num_samples
 
    def __iter__(self):
        batch_indices = []
        for i in range(self.num_samples):
            if (i + 1) * self.batch_size <= self.num_data:
                for idx in range(i * self.batch_size, (i + 1) * self.batch_size):
                    batch_indices.append(self.indices[idx])
                yield batch_indices
                batch_indices = []
            else:
                for idx in range(i * self.batch_size, self.num_data):
                    batch_indices.append(self.indices[idx])
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices
 