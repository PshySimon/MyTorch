from .Optimizer import  Optimizer
import numpy as np

class Adam(Optimizer):
    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(param) for param in self.parameters]
        self.v = [np.zeros_like(param) for param in self.parameters]

    def step(self):
        self.t += 1
        for i, (param, grad) in enumerate(zip(self.parameters, [param.grad for param in self.parameters])):
            if grad is not None:
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
