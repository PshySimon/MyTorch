from nn.module.Module import Module


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ((2 * x).exp() - 1) / ((2 * x).exp() + 1)