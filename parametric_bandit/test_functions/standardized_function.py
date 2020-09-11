import torch
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import Optional, Union


class StandardizedFunction:
    """
    the SyntheticTestFunctions of BoTorch have various bounded domains.
    This class normalizes those to the unit hypercube.
    """

    def __init__(
        self,
        function: SyntheticTestFunction,
        negate: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the function

        Args:
            function: the function to sample from, the initialized object.
            negate: Whether to negate the function output. Many BoTorch test problems
                are intended for minimization, thus the default is True.
            device: The device to run the experiment on. Defaults to `cpu'.
        """
        device = torch.device("cpu") if device is None else device
        self.function = function.to(device)
        self.dim = function.dim
        self.bounds = torch.tensor([[0.0], [1.0]]).repeat(1, self.dim).to(device)
        self.original_bounds = torch.tensor(self.function._bounds).t().to(device)
        self.scale = self.original_bounds[1] - self.original_bounds[0]
        self.l_bounds = self.original_bounds[0]
        self.negate = negate

    def __call__(self, X: Tensor, seed: Optional[int] = None) -> Tensor:
        """
        Scales the solutions to the function domain and evaluates the function

        Args:
            X: Solutions from the relative scale of [0, 1]
            seed: If given, the seed is set for random number generation

        Returns:
            The function value
        """
        old_state = torch.random.get_rng_state()
        if seed is not None:
            torch.random.manual_seed(seed=seed)
        shape = list(X.size())
        shape[-1] = 1
        X = X * self.scale.repeat(shape) + self.l_bounds.repeat(shape)
        result = self.function(X.reshape(-1, X.size(-1))).reshape(shape)
        torch.random.set_rng_state(old_state)
        if self.negate:
            result = -result
        return result

    def evaluate_true(self, X: Tensor) -> Tensor:
        """
        Calls evaluate_true method of the function
        Scales the solutions to the function domain and returns the function value.

        Args:
            X: Solutions from the relative scale of [0, 1]

        Returns:
            The function value
        """
        shape = list(X.size())
        shape[-1] = 1
        X = X * self.scale.repeat(shape) + self.l_bounds.repeat(shape)
        result = self.function.evaluate_true(X).reshape(shape)
        if self.negate:
            result = -result
        return result

    @property
    def optimal_value(self) -> Union[float, None]:
        r"""The global minimum (maximum if negate=True) of the function."""
        if self.function._optimal_value is not None:
            return (
                -self.function._optimal_value
                if self.negate
                else self.function._optimal_value
            )
        else:
            return None
