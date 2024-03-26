import numpy as np
from nn import *
from nn.tests.Common import Timer
import matplotlib.pyplot as plt


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


class Model(Module):
    def __init__(self):
        super().__init__()
        self.ln1 = Linear(1, 64, bias=True)
        self.ln2 = Linear(64, 64, bias=True)
        self.ln3 = Linear(64, 64, bias=True)
        self.ln4 = Linear(64, 1, bias=True)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()
        # self.tanh = Tanh()
        # self.swish = Swish()

    def forward(self, x):
        out = self.ln1(x)
        out = self.relu(out)
        out = self.ln2(out)
        out = self.relu(out)
        out = self.ln3(out)
        out = self.relu(out)
        out = self.ln4(out)
        return out

all_steps = []
all_loss = []
model = Model()
model.train()

grad_clipper = GradClipper(model.parameters())
opt = SGD(parameters=model.parameters(), learning_rate=learning_rate, weight_decay=0.0, decay_type='l2')
loss_fn = MSELoss(reduction="mean")

train_dataset = LinearDataset(X, Y)
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=16, drop_last=True)
for epoch in range(1, epoches):
    loss_total = 0
    for x, y in train_dataloader:
        pred = model(x)
        loss = loss_fn(pred, y)
        loss_total = loss.data
        loss.backward()
        #grad_clipper.clip()
        opt.step()
        opt.clear_grad()
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

with no_grad():
    for x, y in val_dataloader:
        pred = model(x)
        all_pred.append(pred.data)
        all_val.append(x.data)

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
plt.title('numpy model')
plt.legend()

plt.subplot(1, 2, 2)  # 1行2列，第1个子图
plt.plot(all_steps, all_loss, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('numpy model')
plt.legend()

# 显示图形
plt.tight_layout()  # 调整子图布局，防止重叠
plt.show()
