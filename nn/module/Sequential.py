from .Module import Module

class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        self.graphs = []
        self._parameters = []
        for arg_layer in args:
            if isinstance(arg_layer, Module):
                self.graphs.append(arg_layer)
                self._parameters.extend(arg_layer.parameters())
 
    def add(self, layer):
        assert isinstance(layer, Module), "The type of added layer must be Layer, but got {}.".format(type(layer))
        self.graphs.append(layer)
        self._parameters += layer.parameters()
 
    def forward(self, x):
        for graph in self.graphs:
            x = graph(x)
        return x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
 
    def __str__(self):
        string = 'Sequential:\n'
        for graph in self.graphs:
            string += graph.__str__() + '\n'
        return string
 
    def parameters(self):
        return self._parameters