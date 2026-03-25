from torch import nn as nn


class ZeroInitLinear(nn.Linear):
    def __init__(self, channels: int, eps: float = 0.0):
        super().__init__(channels, channels, bias=False)
        # Zero (or tiny-epsilon) init to make the residual branch a no-op at start.
        nn.init.constant_(self.weight, eps)
        self._init_eps = eps
