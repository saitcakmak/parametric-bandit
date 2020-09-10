"""
This is the file to runt he bulk of the experiment from.
We need to make sure that each algorithm gets only the information it needs
and only does the updates it would have done otherwise.
The composite algorithm is the only one which we play around with.
So, OCBA only knows the sample mean and std.
Discrete KG needs to fit GP somehow. We could still fit individual GPs
and then send it a big block covariance matrix to use
"""

import torch
from parametric_bandit.test_functions.function_picker import function_picker
from parametric_bandit.problem import Problem
from parametric_bandit.OCBA import OCBA
from copy import copy
from time import time
from parametric_bandit.discrete_KG import DiscreteKGAlg
from typing import Union

negate = True  # negate the functions for maximization


def OCBA_exp(
    prob: Problem,
    budget: int,
    n: list,
    num_init_samples: int,
    obs_std: float = None,
    randomized: bool = False,
):
    """
    Run the pure OCBA experiment
    :param prob: The initialized Problem object
    :param budget: Total sampling budget after initialization
    :param randomized: If True, use randomized OCBA.
    :param n: list of number of alternatives
    :param num_init_samples: number of samples used for initializing each alternative
    :param obs_std: If given, then OCBA uses known noise level.
    :return: Optimal arm
    """
    ocba = OCBA(
        K=sum(n), N=budget, N_0=num_init_samples, randomized=randomized, maximize=True
    )
    for i in range(budget):
        mean, var = prob.get_arm_stats()
        if obs_std is not None:
            var = torch.tensor([obs_std ** 2] * sum(n))
        next_ = ocba.next_sample(mean, var.pow(1 / 2))
        arm = 0
        while next_ >= n[arm]:
            next_ -= n[arm]
            arm += 1
        if i < budget - 1:
            prob.new_sample(arm, next_, ocba=True)
    mean, var = prob.get_arm_stats()
    best = torch.argmax(mean)
    arm = 0
    while best >= n[arm]:
        best -= n[arm]
        arm += 1
    return arm, best


def KG_exp(prob: Problem, budget: int, obs_std: float, n: list):
    """
    Run the pure KG experiment
    :param prob: The initialized Problem object
    :param budget: Total sampling budget after initialization
    :param obs_std: observation noise level
    :param n: list of number of alternatives per arm
    :return: Optimal arm
    # TODO: in an unknown noise setting, what noise level would we pass to this?
    """
    kg = DiscreteKGAlg(M=sum(n), error=obs_std ** 2, minimize=False)
    for i in range(budget):
        mu, Sigma = prob.get_arm_gp_mean_cov()
        next_ = kg.find_maximizer(mu, Sigma)
        arm = 0
        while next_ >= n[arm]:
            next_ -= n[arm]
            arm += 1
        if i < budget - 1:
            prob.new_sample(arm, next_, ocba=False)
    mu, Sigma = prob.get_arm_gp_mean_cov()
    best = torch.argmax(mu)
    arm = 0
    while best >= n[arm]:
        best -= n[arm]
        arm += 1
    return arm, best


def composite_exp(
    prob: Problem,
    budget: int,
    n: list,
    num_init_samples: int,
    obs_std: float,
    randomized: bool = False,
):
    """
    Run the composite experiment.
    For each arm, when it is chosen, we will do KG to pick which one to sample.
    To pick the arms, we will use OCBA with get_outer_stats for the arm stats.
    :param prob: Problem object
    :param budget: Total sampling budget after initialization
    :param n: number of alternatives in each arm
    :param num_init_samples: number of samples for initialization for each arm
    :param obs_std: For use with KG.
    # TODO: eliminate need for this somehow. Just get the predicted one from the GP fit.
    :param randomized: If True, uses randomized OCBA
    :return: Best arm and alternative
    """
    ocba = OCBA(
        K=len(n), N=budget, N_0=num_init_samples, randomized=randomized, maximize=True
    )
    for i in range(budget):
        mean, std = prob.get_outer_stats()
        next_arm = ocba.next_sample(mean, std)
        kg = DiscreteKGAlg(M=n[next_arm], error=obs_std ** 2, minimize=False)
        mu, Sigma = prob.get_arm_gp_mean_cov(next_arm)
        next_alternative = kg.find_maximizer(mu, Sigma)
        if i < budget - 1:
            prob.new_sample(next_arm, next_alternative, ocba=False)
    mu, Sigma = prob.get_arm_gp_mean_cov()
    best = torch.argmax(mu)
    arm = 0
    while best >= n[arm]:
        best -= n[arm]
        arm += 1
    return arm, best


def function(name: str, obs_std: float):
    return function_picker(function_name=name, noise_std=obs_std, negate=negate)


def single_rep(seed: int, n: list, obs_std: float, N: int, num_init_samples: int):
    """
    Single replication of the experiment
    :param seed: Experiment seed
    :param n: number of samples in each alternative. List of length 3 for now
    :param obs_std: standard deviation of the observations
    :param N: sampling budget after initialization
    :param num_init_samples: number of samples from each alternative for initialization
    :return: regret of all three algorithms
    """
    torch.manual_seed(seed)

    functions = [
        function("branin", obs_std),
        function("levy", obs_std),
        function("hartmann6", obs_std),
        function("ackley", obs_std),
        function("threehumpcamel", obs_std),
        function("sixhumpcamel", obs_std),
    ]
    alternative_points = list()
    for i in range(len(n)):
        alternative_points.append(torch.rand(n[i], functions[i].dim))

    problem = Problem(
        functions=functions, alternative_points=alternative_points, noise_std=obs_std
    )
    problem.initialize_arms(num_samples=num_init_samples)

    # TODO: MPS runs into the cholesky issue, which is likely slowing things down
    # print(problem.mps_test(0, 1000))
    # return 0

    # TODO: adjust it so we can switch between known and unknown noise levels
    if isinstance(num_init_samples, int):
        ocba_best = OCBA_exp(
            prob=copy(problem),
            budget=N,
            n=n,
            num_init_samples=num_init_samples,
            obs_std=obs_std,
        )
        ocba_flat_best = sum(n[: ocba_best[0]]) + ocba_best[1]
    kg_best = KG_exp(prob=copy(problem), budget=N, obs_std=obs_std, n=n)
    kg_flat_best = sum(n[: kg_best[0]]) + kg_best[1]
    if isinstance(num_init_samples, int):
        num_init = n * num_init_samples
    elif isinstance(num_init_samples, list):
        num_init = [len(e) for e in num_init_samples]
    else:
        raise ValueError("num_init_samples is of a type not handled yet.")
    composite_best = composite_exp(
        prob=copy(problem), budget=N, n=n, num_init_samples=num_init, obs_std=obs_std
    )
    composite_flat_best = sum(n[: composite_best[0]]) + composite_best[1]

    val = torch.cat(problem.get_true_val(), dim=0)
    best_val, best = torch.max(val, dim=0)
    arm = 0
    while best >= n[arm]:
        best -= n[arm]
        arm += 1

    if isinstance(num_init_samples, int):
        ocba_regret = best_val - val[ocba_flat_best]
    else:
        ocba_regret = 0.0
    kg_regret = best_val - val[kg_flat_best]
    composite_regret = best_val - val[composite_flat_best]
    return ocba_regret, kg_regret, composite_regret


if __name__ == "__main__":
    start = time()
    res = single_rep(seed=0, n=[10] * 6, obs_std=1.0, N=10, num_init_samples=1)
    print(res)
    print(time() - start)
