from collections.abc import Iterable

class Optimizer:
    """
    optimizer base class.
    Args:
        parameters (Tensor): parameters to be optimized.
        learning_rate (float): learning rate. Default: 0.001.
        weight_decay (float): The decay weight of parameters. Defaylt: 0.0.
        decay_type (str): The type of regularizer. Default: l2.
    """
    def __init__(self, parameters, learning_rate=0.001, weight_decay=0.0, decay_type='l2'):
        assert decay_type in ['l1', 'l2'], "only support decay_type 'l1' and 'l2', but got {}.".format(decay_type)
        if isinstance(parameters, dict):
            self.parameters = list([param for _, param in parameters.items()])
        elif isinstance(parameters, Iterable):
            self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.decay_type = decay_type

    def step(self):
        raise NotImplementedError

    def clear_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def get_decay(self, g):
        if self.decay_type == 'l1':
            return self.weight_decay
        elif self.decay_type == 'l2':
            return self.weight_decay * g
 
