import torch
import numpy as np
from nn.tests.Common import copy_model_weights, compare_weights, Asserter
from nn import Tensor, Module, Linear, ReLU, MSELoss, Adam, DataLoader, Dataset
from torch.utils.data import DataLoader as TorchDataLoader

mid_params1 = []
mid_params2 = []
mid_params1_grad = []
mid_params2_grad = []

class LinearDataset(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y
 
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# 为了读取模型中间参数变量的梯度而定义的辅助函数
def get_grad(grad):
    mid_params2_grad.append(grad)

class Model1(Module):
    def __init__(self):
        super().__init__()
        self.ln1 = Linear(1, 16, bias=True)
        self.ln2 = Linear(16, 64, bias=True)
        self.ln3 = Linear(64, 16, bias=True)
        self.ln4 = Linear(16, 1, bias=True)
        self.relu = ReLU()

    def forward(self, x):
        out = self.ln1(x)
        mid_params1.append(out)
        out = self.relu(out)
        mid_params1.append(out)
        out = self.ln2(out)
        mid_params1.append(out)
        out = self.relu(out)
        mid_params1.append(out)
        out = self.ln3(out)
        mid_params1.append(out)
        out = self.relu(out)
        mid_params1.append(out)
        out = self.ln4(out)
        mid_params1.append(out)
        return out

class Model2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = torch.nn.Linear(1, 16)
        self.ln2 = torch.nn.Linear(16, 64)
        self.ln3 = torch.nn.Linear(64, 16)
        self.ln4 = torch.nn.Linear(16, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        out = self.ln1(x)
        mid_params2.append(out)
        if out.requires_grad:
            out.register_hook(get_grad)

        out = self.relu(out)
        mid_params2.append(out)
        if out.requires_grad:
            out.register_hook(get_grad)

        out = self.ln2(out)
        mid_params2.append(out)
        if out.requires_grad:
            out.register_hook(get_grad)

        out = self.relu(out)
        mid_params2.append(out)
        if out.requires_grad:
            out.register_hook(get_grad)

        out = self.ln3(out)
        mid_params2.append(out)
        if out.requires_grad:
            out.register_hook(get_grad)

        out = self.relu(out)
        mid_params2.append(out)
        if out.requires_grad:
            out.register_hook(get_grad)

        out = self.ln4(out)
        mid_params2.append(out)
        if out.requires_grad:
            out.register_hook(get_grad)
        return out


num_data = 400  # 训练数据数量
val_number = 500    # 验证数据数量
epochs = 200
batch_size = 2
learning_rate = 0.002
X = np.linspace(-np.pi, np.pi, num_data).reshape(num_data, 1)
Y = np.sin(X) * 2 + (np.random.rand(*X.shape) - 0.5) * 0.1
y_ = np.sin(X) * 2

train_dataset1 = LinearDataset(X, Y)
train_dataset2 = LinearDataset(X, Y)
train_dataloader1 = DataLoader(train_dataset1, shuffle=False, batch_size=16, drop_last=True)
train_dataloader2 = TorchDataLoader(train_dataset2, shuffle=False, batch_size=16, drop_last=True)

model1 = Model1()
model2 = Model2()
loss_fn = MSELoss(reduction="mean")
model2.double()
model1.train()
model2.train()
asserter = Asserter()
copy_model_weights(asserter, model1, model2)
opt1 = Adam(parameters=model1.parameters(), learning_rate=0.002)
opt2 = torch.optim.Adam(model2.parameters(), lr=0.002)

def compare_model_weights(asserter, model1, model2):
    for p1, p2 in zip(model1.parameters(), list(model2.parameters())):
        compare_weights(asserter, p1, p2)

def test_model_train(epochs):
    global mid_params1
    global mid_params2
    global mid_params1_grad
    global mid_params2_grad

    compare_model_weights(asserter, model1, model2)

    for i in range(epochs):
        print("\rEpoch {} / {}".format(i, epochs), end="")

        for (x1, y1), (x2, y2) in zip(train_dataloader1, train_dataloader2):
            compare_weights(asserter, x1, x2)
            compare_weights(asserter, y1, y2)
            out1 = model1(x1)
            out2 = model2(x2)
            out2.retain_grad()

            loss1 = loss_fn(out1, y1)
            loss2 = loss_fn(out2, y2)
            loss2.retain_grad()

            asserter( loss1 == loss2, "Mismatched loss value!loss1 = {}, loss2 = {}".format(loss1, loss2))

            loss1.backward()
            loss2.backward()

            compare_weights(asserter, out1, out2)
            compare_weights(asserter, out1.grad, out2.grad)

            compare_weights(asserter, model1.ln1.weight, model2.ln1.weight)
            compare_weights(asserter, model1.ln1.weight.grad, model2.ln1.weight.grad)

            for p1, p2 in zip(mid_params1, mid_params2):
                compare_weights(asserter, p1, p2)

            asserter(len(mid_params1) == len(mid_params2), "Mismatched params num")
            asserter(len(mid_params1_grad) == len(mid_params2_grad), "Mismatched params num")

            mid_params1_grad = [t.grad for t in mid_params1]
            for p1, p2 in zip(mid_params1_grad, mid_params2_grad[::-1]):
                compare_weights(asserter, p1, p2)

            parameters1 = model1.parameters()
            parameters2 = list(model2.parameters())
            for p1, p2 in zip(parameters1, parameters2):
                compare_weights(asserter, p1, p2)
                compare_weights(asserter, p1.grad, p2.grad)

            opt1.step()
            opt1.clear_grad()

            opt2.step()
            opt2.zero_grad()

            mid_params1 = []
            mid_params2 = []
            mid_params1_grad = []
            mid_params2_grad = []

            compare_model_weights(asserter, model1, model2)

test_model_train(epochs)
asserter.report()

for p1, p2 in zip(model1.parameters(), list(model2.parameters())):
    compare_weights(asserter, p1, p2)

import matplotlib.pyplot as plt
X_val = np.linspace(-np.pi, np.pi, val_number).reshape(val_number, 1)
Y_val = np.sin(X_val) * 2
val_dataset1 = LinearDataset(X_val, Y_val)
val_dataloader1 = DataLoader(val_dataset1, shuffle=False, batch_size=2, drop_last=False)
from nn import no_grad
all_pred1 = []
all_val1 = []
model1.eval()

with no_grad():
    for x, y in val_dataloader1:
        pred = model1(x)
        all_pred1.append(pred.data)
        all_val1.append(x.data)

all_pred1 = np.vstack(all_pred1)
all_val1 = np.vstack(all_val1)

val_dataset2 = LinearDataset(X_val, Y_val)
val_dataloader2 = TorchDataLoader(val_dataset2, shuffle=False, batch_size=2, drop_last=False)
all_pred2 = []
all_val2 = []
model2.eval()

with torch.no_grad():
    for x, y in val_dataloader2:
        pred = model2(x)
        all_pred2.append(pred.data)
        all_val2.append(x.data)

all_pred2 = np.vstack(all_pred2)
all_val2 = np.vstack(all_val2)

for pred1, pred2 in zip(all_pred1, all_pred2):
    compare_weights(asserter, pred1, pred2)

plt.scatter(X, Y, marker='x')
plt.plot(all_val1, all_pred1, color='red')
#plt.plot(all_val2, all_pred2, color='blue')
plt.show()


