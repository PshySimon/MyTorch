import numpy as np
from nn.module.Module import Module


class ReLU(Module):
    """
    forward formula:
        relu = x if x >= 0
             = 0 if x < 0
    backwawrd formula:
        grad = gradient * (x > 0)
    """
    def __init__(self, isTraining=True):
        super().__init__(isTraining=isTraining)
 
    def forward(self, x):
        return x.relu()
    
    def parameters(self):
        return {}
