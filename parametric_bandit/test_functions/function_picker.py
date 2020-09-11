"""
The function picker.
Find most in: https://www.sfu.ca/~ssurjano/optimization.html
"""
from parametric_bandit.test_functions.standardized_function import StandardizedFunction
from botorch.test_functions import (
    Ackley,
    Beale,
    Branin,
    DixonPrice,
    EggHolder,
    Levy,
    Rastrigin,
    SixHumpCamel,
    ThreeHumpCamel,
    Powell,
    Hartmann,
    Rosenbrock,
)
from parametric_bandit.test_functions.other_synthetic_functions import Alpine
import torch
from typing import Optional

function_dict = {
    "powell": Powell,
    "beale": Beale,
    "dixonprice": DixonPrice,
    "eggholder": EggHolder,
    "levy": Levy,
    "rastrigin": Rastrigin,
    "branin": Branin,
    "ackley": Ackley,
    "hartmann": Hartmann,
    "sixhumpcamel": SixHumpCamel,
    "threehumpcamel": ThreeHumpCamel,
    "alpine": Alpine,
    "rosenbrock": Rosenbrock,
}


def function_picker(
    function_name: str,
    noise_std: float,
    negate: bool = True,
    device: Optional[torch.device] = None,
) -> StandardizedFunction:
    """
    Returns the appropriate function callable
    If adding new BoTorch test functions, run them through StandardizedFunction.
    StandardizedFunction and all others listed here allow for a seed to be specified.
    If adding something else, make sure the forward (or __call__) takes a seed argument.

    Args:
        function_name: Function to be used. If the last character is a digit,
            it is used as the dimension of the function. Passing a dimension with a
            function that does not accept dimension may raise an Exception.
        noise_std: the standard deviation of the observation noise
        negate: In most cases, should be true for maximization. This is passed to the
            StandardizedFunction, not the function itself, to ensure that evaluate_true
            is also negated.
        device: The device to run the experiment on. Defaults to `cpu'.

    Returns:
        The function callable
    """
    if function_name[-1].isdigit():
        dim = int(function_name[-1])
        function_name = function_name[:-1]
    else:
        dim = None
    if function_name in function_dict.keys():
        # if the last character of function name is a number, then it is used as the
        # dimension of the function.
        if dim is not None:
            function = StandardizedFunction(
                function_dict[function_name](dim=dim, noise_std=noise_std),
                negate=negate,
                device=device,
            )
        else:
            function = StandardizedFunction(
                function_dict[function_name](noise_std=noise_std),
                negate=negate,
                device=device,
            )
    else:
        raise ValueError("Function name was not found!")
    return function
