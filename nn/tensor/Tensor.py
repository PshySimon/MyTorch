import torch
import numpy as np

np.seterr(divide='ignore', invalid='ignore')

no_grad_ = False

def get_value(param):
    if isinstance(param, Tensor):
        return param.data
    return param


def find_broad_cast_dim(tensor, grad):
    if isinstance(tensor, float) or isinstance(tensor, int):
        return -1
    if not isinstance(tensor.data, np.ndarray) or not isinstance(grad, np.ndarray):
        return -1
    if len(grad.shape) - len(tensor.shape()) == 1:
        return 0
    if len(grad.shape) != len(tensor.shape()):
        return -1
    dims = -1
    for i, x, y in zip(list(range(len(tensor.shape()))), tensor.shape(), grad.shape):
        if x != y and dims == -1:
            return i
    return dims

class GradFn:
    def __init__(self, parent1, parent2):
        self.parent1 = parent1
        self.parent2 = parent2
        if parent1 is not None and isinstance(parent1, Tensor):
            parent1.grad = np.zeros_like(parent1.data)
        if parent2 is not None and isinstance(parent2, Tensor):
            parent2.grad = np.zeros_like(parent2.data)

    def caculate(self, param, grad):
        if isinstance(param, Tensor):
            param.backward(grad)


class Add(GradFn):
    def __init__(self, parent1, parent2):
        super().__init__(parent1, parent2)
    
    def __call__(self, grad):
        # 广播机制，如果A + B形状不一致，如：A=[16, 64], B=[1, 64]
        # 那么就沿着广播的维度求和后再处理
        dims1 = find_broad_cast_dim(self.parent1, grad)
        dims2 = find_broad_cast_dim(self.parent2, grad)
        self.caculate(self.parent1, grad if dims1 == -1 else np.sum(grad, axis=dims1))
        self.caculate(self.parent2, grad if dims2 == -1 else np.sum(grad, axis=dims2))


class ReLU(GradFn):
    def __init__(self, parent1, parent2):
        super().__init__(parent1, parent2)

    def __call__(self, grad):
        self.caculate(self.parent1, self.parent2 * grad)


class Exp(GradFn):
    def __init__(self, parent1):
        super().__init__(parent1, parent2=None)

    def __call__(self, grad):
        # y = e^x
        # d_y_d_x = e^x
        self.caculate(self.parent1, np.exp(self.parent1.data) * grad)

class Sum(GradFn):
    def __init__(self, parent1):
        super().__init__(parent1, parent2=None)
    
    def __call__(self, grad):
        self.caculate(self.parent1, np.ones_like(self.parent1.data))


class Mean(GradFn):
    def __init__(self, parent1):
        super().__init__(parent1, parent2=None)

    def __call__(self, grad):
        self.caculate(self.parent1, np.ones_like(self.parent1.data) / self.parent1.data.size)


class Sub(GradFn):
    def __init__(self, parent1, parent2):
        super().__init__(parent1, parent2)
    
    def __call__(self, grad):
        self.caculate(self.parent1, grad)
        self.caculate(self.parent2, -1. * grad)


class Mul(GradFn):
    def __init__(self, parent1, parent2):
        super().__init__(parent1, parent2)
    
    def __call__(self, grad):
        # z = x * y
        # d_z / d_x = y
        # d_z / d_y = x
        parent1 = get_value(self.parent1)
        parent2 = get_value(self.parent2)
        d_z_d_x = parent2
        d_z_d_y = parent1
        self.caculate(self.parent1, d_z_d_x * grad)
        self.caculate(self.parent2, d_z_d_y * grad)


class Div(GradFn):
    def __init__(self, parent1, parent2):
        super().__init__(parent1, parent2)
    
    def __call__(self, grad):
        # z = x / y
        # d_z / d_x = 1 / y
        # d_z / d_y = x * (-(1 / y^2))
        parent1 = get_value(self.parent1)
        parent2 = get_value(self.parent2)
        d_z_d_x = 1. / parent2
        d_z_d_y = parent1 * (-(1. / (parent2 ** 2)))
        self.caculate(self.parent1, d_z_d_x * grad)
        self.caculate(self.parent2, d_z_d_y * grad)


class Pow(GradFn):
    def __init__(self, parent1, parent2):
        super().__init__(parent1, parent2)
    
    def __call__(self, grad):
        # z = x^y
        # d_z / d_x = y * x ^ (y - 1)
        # d_z / d_y = x ^ y * ln(x)
        parent1 = get_value(self.parent1)
        parent2 = get_value(self.parent2)
        d_z_d_x = parent2 * parent1 ** (parent2 - 1)
        if isinstance(parent2, Tensor):
            d_z_d_y = parent1 ** parent2 * np.log(parent1)
            self.caculate(self.parent2, d_z_d_y * grad)
        self.caculate(self.parent1, d_z_d_x * grad)


class MatMul(GradFn):
    def __init__(self, parent1, parent2):
        super().__init__(parent1, parent2)

    def __call__(self, grad):
        # z = x @ y
        # d_z_d_x = y^T
        # d_z_d_y = x
        # x = [[x1, x2, x3], [x4, x5, x6]] y = [[y1, y2], [y3, y4], [y5, y6]]
        # z = [[x1y1+x2y3+x3y5, x1y2+x2y4+x3y6], [x4y1+x5y3+x6y5, x4y2+x5y4+x6y6]]
        # d_z / d_x = [[y1, y4, y5+y6], [y1+y2, y3+y4, y5+y6]]
        # d_z / d_y = [[x1+x4, x1+x4], [x2+x5, x2+x5], [x3+x6, x3+x6]]
        parent1 = get_value(self.parent1)
        parent2 = get_value(self.parent2)
        d_z_d_x = parent2.T
        d_z_d_y = parent1.T
        self.caculate(self.parent1, np.dot(grad, d_z_d_x))
        self.caculate(self.parent2, np.dot(d_z_d_y, grad))


class Tensor:
    def __init__(self, data, grad=None, name=None, requires_grad=False):
        if isinstance(data, list):
            self.data = np.array(data)
        else:
            self.data = data
        self.requires_grad = requires_grad
        if requires_grad:
            self.grad = np.ones_like(data)
        else:
            self.grad = grad
        self.grad_fn = None
        self.name = name

    def get_value(self, other):
        if isinstance(other, Tensor):
            return other.data
        return other
    
    def no_grad_(self):
        global no_grad_
        return no_grad_
    
    def __add__(self, other):
        value = get_value(other)
        result = Tensor(self.data + value)
        if not self.no_grad_() and (self.requires_grad or
                                    (isinstance(other, Tensor) and other.requires_grad)):
            result.requires_grad = True
            result.grad_fn = Add(self, other)
        return result
    
    def __radd__(self, other):
        result = Tensor(self.data + other)
        if not self.no_grad_() and self.requires_grad:
            result.requires_grad = True
            result.grad_fn = Add(other, self)
        return result
    
    def __sub__(self, other):
        value = get_value(other)
        result = Tensor(self.data - value)
        if not self.no_grad_() and (self.requires_grad or other.requires_grad):
            result.requires_grad = True
            result.grad_fn = Sub(self, other)
        return result
    
    def __rsub__(self, other):
        result = Tensor(other - self.data)
        if not self.no_grad_() and self.requires_grad:
            result.requires_grad = True
            result.grad_fn = Sub(other, self)
        return result
    
    def __mul__(self, other):
        value = get_value(other)
        result = Tensor(self.data * value)
        if not self.no_grad_() and (self.requires_grad or
                                    (isinstance(other, Tensor) and other.requires_grad)):
            result.requires_grad = True
            result.grad_fn = Mul(self, other)
        return result
    
    def __rmul__(self, other):
        result = Tensor(other * self.data)
        if not self.no_grad_() and self.requires_grad:
            result.requires_grad = True
            result.grad_fn = Mul(other, self)
        return result
    
    def __truediv__(self, other):
        value = get_value(other)
        result = Tensor(self.data / value)
        if not self.no_grad_() and (self.requires_grad or
                                    (isinstance(other, Tensor) and other.requires_grad)):
            result.requires_grad = True
            result.grad_fn = Div(self, other)
        return result
    
    def __rtruediv__(self, other):
        result = Tensor(other / self.data)
        if not self.no_grad_() and self.requires_grad:
            result.requires_grad = True
            result.grad_fn = Div(other, self)
        return result
    
    def __pow__(self, other):
        value = get_value(other)
        result = Tensor(np.power(self.data, value))
        if not self.no_grad_() and (self.requires_grad or
                                    (isinstance(other, Tensor) and other.requires_grad)):
            result.requires_grad = True
            result.grad_fn = Pow(self, other)
        return result
    
    def __rpow__(self, other):
        result = Tensor(np.power(other, self.data))
        if not self.no_grad_() and self.requires_grad:
            result.requires_grad = True
            result.grad_fn = Pow(other, self)
        return result
    
    def __matmul__(self, other):
        value = get_value(other)
        result = Tensor(np.dot(self.data, value))
        if not self.no_grad_() and (self.requires_grad or other.requires_grad):
            result.requires_grad = True
            result.grad_fn = MatMul(self, other)
        return result

    def relu(self):
        tmp = np.copy(self.data)
        tmp[tmp < 0] = 0
        result = Tensor(tmp)
        if not self.no_grad_() and self.requires_grad:
            result.requires_grad = True
            result.grad_fn = ReLU(self, self.data > 0)
        return result

    def exp(self):
        result = Tensor(np.exp(self.data))
        if not self.no_grad_() and self.requires_grad:
            result.requires_grad = True
            result.grad_fn = Exp(self)
        return result
    
    def sum(self):
        result = Tensor(np.sum(self.data))
        if not self.no_grad_() and self.requires_grad:
            result.requires_grad = True
            result.grad_fn = Sum(self)
        return result

    def mean(self):
        result = Tensor(np.mean(self.data))
        if not self.no_grad_() and self.requires_grad:
            result.requires_grad = True
            result.grad_fn = Mean(self)
        return result
    
    def __eq__(self, other):
        if isinstance(other, list):
            return np.allclose(self.data, np.array(other))
        elif isinstance(other, torch.Tensor):
            return np.allclose(self.data, other.detach().numpy())
        else:
            return np.allclose(self.data, other)
    
    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is not None and grad.dtype != np.float64:
            raise TypeError("Unexpected type, please check input")
        if grad is None:
            grad = np.ones_like(self.data)

        if self.grad is None:
            self.grad = grad
        else:
            self.grad += grad
        
        if self.grad_fn is not None:
            self.grad_fn(self.grad)
    
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
    
    def numpy(self):
        return np.array(self.data)

    def transpose(self):
        self.data = self.data.T
        self.grad = self.grad.T
        return self

    def shape(self):
        if isinstance(self.data, int or float):
            return (1,)
        elif isinstance(self.data, np.ndarray):
            return self.data.shape
        return None

    def __repr__(self):
        return f'{np.array(self.data)}'


class no_grad:
    def __enter__(self):
        global no_grad_
        no_grad_ = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        global no_grad_
        no_grad_ = False

    