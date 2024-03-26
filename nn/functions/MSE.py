from ..module.Module import Module


class MSELoss(Module):
    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, y_pred, label):
        out = 0.5 * (y_pred - label) ** 2
        if self.reduction == "none":
            return out
        elif self.reduction == "sum":
            return out.sum()
        elif self.reduction == "mean":
            return out.mean()
        else:
            raise NotImplementedError("Unimplemented reduction method of `MSE`")


