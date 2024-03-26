import numpy as np
from .Optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum
        self.velocity = []
        for param in self.parameters:
            self.velocity.append(np.zeros_like(param.grad))
 
    def step(self):
        for p, v in zip(self.parameters, self.velocity):
            decay = self.get_decay(p.grad)
            v = self.momentum * v + p.grad + decay # 动量计算
            p.data = p.data - self.learning_rate * v