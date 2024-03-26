import numpy as np
from nn.module.Module import Module
from nn.tensor.Tensor import Tensor


class Linear(Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = Tensor(
            np.random.uniform(low=-np.sqrt(1/in_features),
                              high=np.sqrt(1/in_features),
                              size=(in_features, out_features)),
            requires_grad=True,
            name="Linear weight"
        )
        self.bias_ = bias
        if self.bias_:
            self.bias = Tensor(np.zeros(out_features),
                                requires_grad=True)

    def forward(self, X):
        out = X @ self.weight
        if self.bias_:
            out = out + self.bias
        return out

    def named_parameters(self):
        named_parameters = {"weight": self.weight}
        if self.bias:
            named_parameters["bias"] = self.bias
        return named_parameters
    
    def parameters(self):
        parameters = [self.weight]
        if self.bias_:
            parameters.append(self.bias)
        return parameters
            
    
