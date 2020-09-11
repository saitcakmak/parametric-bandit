"""
Any additional synthetic test functions can be defined here.
"""
import torch
from torch import Tensor
from botorch.test_functions import SyntheticTestFunction
from typing import Optional
import math


class Alpine(SyntheticTestFunction):
    """
    The Alpine-2 test function.

    n-dimensional function typically evaluated on x_i in [0, 10].

    A(x) = - prod_{i=1}^n sqrt(x_i) sin(x_i).
        (negated to make it into a minimization problem by default)

    The global optimum is found at x_i ≈ 7.91705268466621...
    """

    def __init__(
        self, dim=6, noise_std: Optional[float] = None, negate: bool = False
    ) -> None:
        self.dim = dim
        self._bounds = [(0.0, 10.0) for _ in range(self.dim)]
        self._optimizers = [tuple(7.91705268466621 for _ in range(self.dim))]
        self._optimal_value = -math.pow(2.808130979537964, self.dim)
        super().__init__(noise_std=noise_std, negate=negate)

    def evaluate_true(self, X: Tensor) -> Tensor:
        return -torch.prod(torch.sqrt(X) * torch.sin(X), dim=-1, keepdim=True)
