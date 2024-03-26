import torch
import numpy as np
from nn import Tensor


def check_grad(tensor1, tensor2):
    return np.allclose(tensor1.grad, tensor2.grad.detach().numpy())

def zero_grad(tensor1, tensor2):
    tensor1.zero_grad()
    if tensor2.grad is not None:
        tensor2.grad.zero_()

def test_forward_same_shape_value():
    TestCases = [
        (np.random.rand(3, 3), np.random.rand(3, 3)),
        (np.random.rand(3, 2), np.random.rand(3, 2)),
        (np.random.rand(15, 2), np.random.rand(15, 2)),
    ]

    for op1, op2 in TestCases:
        myTensor1 = Tensor(op1, requires_grad=True)
        myTensor2 = Tensor(op2, requires_grad=True)
        torchTensor1 = torch.tensor(op1, requires_grad=True)
        torchTensor2 = torch.tensor(op2, requires_grad=True)

        myAdd = myTensor1 - myTensor2
        torchAdd = torchTensor1 - torchTensor2
        assert myAdd == torchAdd, "Addition comarasion failed"

        myAdd.backward()
        torchAdd.retain_grad()
        torchAdd.sum().backward()
        assert check_grad(myTensor1, torchTensor1), "Addition grad comparasion failed"
        assert check_grad(myTensor2, torchTensor2), "Addition grad comparasion failed"
        zero_grad(myAdd, torchAdd)

        # 减法
        mySub = myTensor1 - myTensor2
        torchSub = torchTensor1 - torchTensor2
        assert mySub == torchSub, "Subtraction comparison failed"

        mySub.backward()
        torchSub.retain_grad()
        torchSub.sum().backward()
        assert check_grad(myTensor1, torchTensor1), "Subtraction grad comparasion failed"
        assert check_grad(myTensor2, torchTensor2), "Subtraction grad comparasion failed"
        zero_grad(mySub, torchSub)

        # 乘法
        myMul = myTensor1 * myTensor2
        torchMul = torchTensor1 * torchTensor2
        assert myMul == torchMul, "Multiplication comparison failed"

        myMul.backward()
        torchMul.retain_grad()
        torchMul.sum().backward()
        assert check_grad(myTensor1, torchTensor1), "Multiplication grad comparasion failed"
        assert check_grad(myTensor2, torchTensor2), "Multiplication grad comparasion failed"
        zero_grad(myMul, torchMul)

        # 除法
        myDiv = myTensor1 / myTensor2
        torchDiv = torchTensor1 / torchTensor2
        assert myDiv == torchDiv, "Division comparison failed"

        myDiv.backward()
        torchDiv.retain_grad()
        torchDiv.sum().backward()
        assert check_grad(myTensor1, torchTensor1), "Division grad comparasion failed"
        assert check_grad(myTensor2, torchTensor2), "Division grad comparasion failed"
        zero_grad(myDiv, torchDiv)

        # 幂运算
        myPow = myTensor1 ** myTensor2
        torchPow = torch.pow(torchTensor1, torchTensor2)
        assert myPow == torchPow, "Exponentiation comparison failed"

        myPow.backward()
        torchPow.retain_grad()
        torchPow.sum().backward()
        assert check_grad(myTensor1, torchTensor1), "Exponentiation grad comparasion failed"
        assert check_grad(myTensor2, torchTensor2), "Exponentiation grad comparasion failed"
        zero_grad(myPow, torchPow)


def test_forward_matrix_value():
    TestCases = [
        (np.random.rand(3, 2), np.random.rand(2, 3)),
        (np.random.rand(2, 2), np.random.rand(2, 2)),
        (np.random.rand(15, 2), np.random.rand(2, 7)),
    ]

    for op1, op2 in TestCases:
        myTensor1 = Tensor(op1, requires_grad=True)
        myTensor2 = Tensor(op2, requires_grad=True)
        torchTensor1 = torch.tensor(op1, requires_grad=True)
        torchTensor2 = torch.tensor(op2, requires_grad=True)

        # 矩阵乘法
        myMatMul = myTensor1 @ myTensor2
        torchMatMul = torchTensor1 @ torchTensor2
        assert myMatMul == torchMatMul, "Matrix multiplication comparison failed"

        myMatMul.backward()
        torchMatMul.retain_grad()
        torchMatMul.sum().backward()
        assert check_grad(myTensor1, torchTensor1), "Matrix multiplication grad comparasion failed"
        assert check_grad(myTensor2, torchTensor2), "Matrix multiplication grad comparasion failed"
        zero_grad(myMatMul, torchMatMul)


if __name__ == "__main__":
    test_forward_matrix_value()
    test_forward_same_shape_value()