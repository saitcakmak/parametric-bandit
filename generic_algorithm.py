from typing import Callable, List

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction

from arm import ParametricArm


class GenericAlgorithm:
    """
    The generic algorithm for the parametric bandit defined in the notes
    It does not specify g() and h() functions
    This doesn't do too well. It has many shortcomings.
    """

    def __init__(self, arm_functions: List[SyntheticTestFunction], h: Callable, g: Callable, w_star0: float,
                 n0: int = 10, retrain_gp: bool = False, verbose: bool = True):
        """
        Initialize the algorithm set-up with the given number of arms

        :param arm_functions: list of functions of each arm
        :param h: the function for allocating the weight between inferior arms
        :param g: the function for setting the weight of the best arm
        :param w_star0: the initial weight of the best arm
        :param n0: number of initial samples for each arm
        :param retrain_gp: if True, the arm models are retrained after each sample
        :param verbose: if True, print some progress messages
        """
        self.verbose = verbose
        self._initialize_arms(arm_functions, n0, retrain_gp)
        self.initial_kg = self.arm_kg.clone()
        self.h = h
        self.g = g
        self.w_star0 = w_star0
        self.N = 0
        if self.verbose:
            print("Initialization complete. mu: %s, KG: %s" % (self.arm_mu_best, self.arm_kg))

    def _initialize_arms(self, arm_functions, n0, retrain_gp):
        """
        Initialize the arms as ParametricArm objects.
        Record the current mu_best and KG values

        :param arm_functions: The test functions for each arm
        :param n0: number of initial samples
        :return: None
        """
        self.arms = []
        self.arm_mu_best = torch.empty(len(arm_functions))
        self.arm_kg = torch.empty(len(arm_functions))
        for i in range(len(arm_functions)):
            arm = ParametricArm(function=arm_functions[i], num_init_samples=n0, retrain_gp=retrain_gp)
            self.arms.append(arm)
            self.arm_mu_best[i] = arm.current_best_val
            self.arm_kg[i] = arm.kg_value
            if self.verbose:
                print("Arm %d is initialized." % i)

    def sample_next(self):
        """
        Sample the next arm suggested by the algorithm
        :return:
        """
        # calculate weights w_n
        best_mu, best_index = torch.max(self.arm_mu_best, 0)
        K_0 = self.initial_kg.sum() - self.initial_kg[best_index]
        K_n = self.arm_kg.sum = self.arm_kg[best_index]
        w_star = self.w_star0 + (1 - self.w_star0) * (1 - self.g(K_n / K_0))
        delta = self.arm_mu_best - best_mu
        bar_delta = delta / delta.sum()
        h_bar_delta = self.h(bar_delta)
        # TODO: w_n returns values outside the range [0,1] check what is wrong here - this is because of negative KG
        #   we also get negative KG values, figure out how to handle that - maybe analytic EI, that seems to work
        normalized_h = h_bar_delta / (h_bar_delta.sum() - h_bar_delta[best_index])
        w_n = (1 - w_star) * normalized_h
        w_n[best_index] = w_star
        # next sample = arg max w_n * KG
        next_sample = torch.argmax(w_n * self.arm_kg, 0)
        if self.verbose:
            print("w_n = ", w_n)
            print("kg = ", self.arm_kg)
            print("ww = ", w_n * self.arm_kg)
        # take the next sample and update relevant values
        if self.verbose:
            print("Sampling from arm %d." % int(next_sample))
        self.arms[int(next_sample)].sample_next()
        self.arm_kg[next_sample] = self.arms[int(next_sample)].kg_value
        self.arm_mu_best[next_sample] = self.arms[int(next_sample)].current_best_val
        self.N += 1
        if self.verbose:
            print("%d th sample is complete. New mu, kg: (%s, %s)" % (self.N, self.arm_mu_best, self.arm_kg))

