from ..tensor.Tensor import Tensor

class Module:
    def __init__(self, isTraining=False):
        self.isTraining = isTraining
        self._parameters = []
        self._named_parameters = {}
        self._sub_modules = []
 
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def named_parameters(self):
        if not self._named_parameters:
            for attr in dir(self):
                obj = getattr(self, attr)
                if isinstance(obj, Module):
                    self._sub_modules.append(obj)
                    for name, param in obj.named_parameters().items():
                        self._named_parameters["{}.{}".format(attr, param)] = param
        return self._named_parameters
 
    def parameters(self):
        if not self._parameters:
            named_parameters = self.named_parameters()
            self._parameters = [v for _, v in named_parameters.items()]
        return self._parameters
 
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_init_modules(self):
        if not self._sub_modules:
            for attr in dir(self):
                obj = getattr(self, attr)
                if isinstance(obj, Module):
                    self._sub_modules.append(obj)
    def train(self):
        self.get_init_modules()
        for module in self._sub_modules:
            module.isTraining = True

    def eval(self):
        self.get_init_modules()
        for module in self._sub_modules:
            module.isTraining = False

