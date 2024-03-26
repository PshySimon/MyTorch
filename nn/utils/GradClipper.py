import numpy as np

class GradClipper:
    def __init__(self, parameters, max_grad_norm=1.0):
        self.parameters = parameters
        self.max_grad_norm = max_grad_norm

    def clip(self):
        # 计算梯度范数
        total_norm = 0.0
        for param in self.parameters:
            param_norm = np.linalg.norm(param.grad)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5  # 计算总的梯度范数

        # 计算裁剪比例
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)

        # 如果裁剪比例小于1，则对梯度进行裁剪
        if clip_coef < 1:
            for param in self.parameters:
                param.grad = param.grad * clip_coef  # 将梯度乘以裁剪比例

