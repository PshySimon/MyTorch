from nn.module.Module import Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1. / (1. + (-1. * x).exp())