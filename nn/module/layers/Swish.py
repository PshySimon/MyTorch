from nn.module.Module import Module


class Swish(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x / (1. + (-1. * x).exp())