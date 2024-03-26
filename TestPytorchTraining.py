import torch
import torch.nn as nn
import numpy as np
from nn.tests.Common import Timer
from nn import Dataset, BatchSampler, MSELoss
import matplotlib.pyplot as plt
from torch.optim import SGD


class Optimizer:
    def __init__(self, parameters, learning_rate=0.001, weight_decay=0.0, decay_type='l2'):
        assert decay_type in ['l1', 'l2'], "only support decay_type 'l1' and 'l2', but got {}.".format(decay_type)
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.decay_type = decay_type

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None

    def get_decay(self, g):
        if self.decay_type == 'l1':
            return self.weight_decay
        elif self.decay_type == 'l2':
            return self.weight_decay * g


class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [torch.zeros_like(param) for param in self.parameters]
        self.v = [torch.zeros_like(param) for param in self.parameters]

    def step(self):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.parameters, [param.grad for param in self.parameters])):
            if grad is not None:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)



class MySGD(Optimizer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum
        self.velocity = []
        for param in self.parameters:
            self.velocity.append(np.zeros_like(param.grad))

    def step(self):
        for p, v in zip(self.parameters, self.velocity):
            decay = self.get_decay(p.grad)
            v = self.momentum * v + p.grad + decay  # 动量计算
            p.data = p.data - self.learning_rate * v

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
            yield torch.tensor(np.stack(data_list, axis=0)), torch.tensor(np.stack(label_list, axis=0))


class LinearDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


timer = Timer()
timer.start()
num_data = 300  # 训练数据数量
val_number = 500    # 验证数据数量
epoches = 1500
batch_size = 16
learning_rate = 0.005

X = np.linspace(-np.pi, np.pi, num_data).reshape(num_data, 1)
Y = np.sin(X) * 2 + (np.random.rand(*X.shape) - 0.5) * 0.1
y_ = np.sin(X) * 2


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.Linear(1, 64)
        self.ln2 = nn.Linear(64, 64)
        self.ln3 = nn.Linear(64, 64)
        self.ln4 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.ln1(x)
        out = self.relu(out)
        out = self.ln2(out)
        out = self.relu(out)
        out = self.ln3(out)
        out = self.relu(out)
        out = self.ln4(out)
        return out


model = Model()
model.double()
model.train()

opt = MySGD(parameters=model.parameters(), learning_rate=learning_rate, weight_decay=0.0)
# opt = SGD(model.parameters(), lr=learning_rate, weight_decay=0.0)
# opt = Adam(parameters=model.parameters(), learning_rate=learning_rate, weight_decay=0.0, decay_type='l2')
loss_fn = MSELoss(reduction="mean")

all_steps = []
all_loss = []
train_dataset = LinearDataset(X, Y)
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=16, drop_last=True)
for epoch in range(1, epoches):
    loss_total = 0
    for x, y in train_dataloader:
        pred = model(x)
        loss = loss_fn(pred, y)
        loss_total = loss.data
        loss.backward()
        opt.step()
        opt.zero_grad()
    print("epoch: {}. loss: {}".format(epoch, loss_total))
    all_steps.append(epoch)
    all_loss.append(loss_total)

X_val = np.linspace(-np.pi, np.pi, val_number).reshape(val_number, 1)
Y_val = np.sin(X_val) * 2
val_dataset = LinearDataset(X_val, Y_val)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=2, drop_last=False)
all_pred = []
all_val = []
model.eval()
for x, y in val_dataloader:
    pred = model(x)
    all_pred.append(pred.detach().numpy())
    all_val.append(x.detach().numpy())

all_pred = np.vstack(all_pred)
all_val = np.vstack(all_val)

timer.stop()
timer.report()


plt.figure(figsize=(10, 5))  # 设置图形大小

# 第一幅图
plt.subplot(1, 2, 1)  # 1行2列，第1个子图
plt.scatter(X, Y, marker='x')
plt.plot(all_val, all_pred, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('pytorch model')
plt.legend()

plt.subplot(1, 2, 2)  # 1行2列，第1个子图
plt.plot(all_steps, all_loss, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('pytorch model')
plt.legend()

# 显示图形
plt.tight_layout()  # 调整子图布局，防止重叠
plt.show()