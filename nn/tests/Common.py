import time
import torch
import numpy as np
from nn import Tensor, Module

def shape(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.size()
    elif isinstance(tensor, Tensor):
        return tensor.shape()
    elif isinstance(tensor, np.ndarray):
        return tensor.shape
    else:
        raise NotImplementedError

def copy_tensor_weights(tensor1: Tensor, tensor2: torch.nn.Parameter):
    if shape(tensor1) == shape(tensor2):
        tensor1.data = tensor2.data.detach().numpy()
    elif shape(tensor1) == shape(tensor2.data.T):
        tensor1.data = tensor2.data.detach().numpy().T
    else:
        raise ValueError("Mismatch tensor shape, cannot copy weight")

def compare_weights(asserter, tensor1: Tensor, tensor2: torch.Tensor):
    if tensor1 is None or tensor2 is None:
        raise ValueError
    if tensor1 is None and tensor2 is None:
        return
    if shape(tensor1) == shape(tensor2):
        if isinstance(tensor1, np.ndarray) and isinstance(tensor2, np.ndarray):
            asserter(np.allclose(tensor1.data, tensor2),
                        "Mismatched weights tensor1 {}, tensor2 {}".format(tensor1, tensor2))
        else:
            asserter( np.allclose(tensor1.data, tensor2.detach().numpy()),
                         "Mismatched weights tensor1 {}, tensor2 {}".format(tensor1, tensor2))
    elif shape(tensor1) == shape(tensor2.data.T):
        if isinstance(tensor1, np.ndarray):
            asserter( np.allclose(tensor1, tensor2.detach().numpy().T), "Mismatched weights")
        else:
            asserter( np.allclose(tensor1.data, tensor2.detach().numpy().T), "Mismatched weights")
    else:
        raise ValueError("Mismatch tensor shape, cannot compare weight")

def copy_model_weights(asserter, model1: Module, model2: torch.nn.Module):
    parameters1 = model1.parameters()
    parameters2 = list(model2.parameters())
    if len(parameters1) != len(parameters2):
        raise ValueError("Model parameter num mismatched!")

    for p1, p2 in zip(parameters1, parameters2):
        copy_tensor_weights(p1, p2)
        compare_weights(asserter, p1, p2)


class Asserter:
    def __init__(self):
        self.pass_count = 0
        self.fail_count = 0

    def __call__(self, condition, message=None):
        try:
            assert condition, message
            self.pass_count += 1
        except AssertionError:
            self.fail_count += 1

    def report(self):
        print("\nPass: {}, Fail: {}".format(self.pass_count, self.fail_count))

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def report(self):
        if self.start_time is None:
            print("Timer hasn't started yet.")
            return
        if self.end_time is None:
            print("Timer hasn't stopped yet.")
            return

        elapsed_time = self.end_time - self.start_time
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
