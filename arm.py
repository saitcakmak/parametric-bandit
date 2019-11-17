import torch
from torch import Tensor
from botorch.acquisition import qKnowledgeGradient, PosteriorMean, qExpectedImprovement, ExpectedImprovement
from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import Optional, Union, Tuple
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from botorch.optim import optimize_acqf


# noinspection PyArgumentList
class ParametricArm:
    """
    the class of an Arm
    """

    def __init__(self, function: SyntheticTestFunction,
                 num_init_samples: int = 10, retrain_gp: bool = False,
                 num_restarts: int = 10, raw_samples: int = 1000):
        """
        Initialize the Arm

        :param function: the function of the arm to sample from
        :param num_init_samples: number of samples to initialize with
        :param retrain_gp: retrain the model after each sample if True
        :param num_restarts: number of random restarts for acquisition function optimization
        :param raw_samples: number of raw samples for acquisition function optimization
        """
        self.function = function
        self.dim = function.dim
        self.bounds = Tensor(function._bounds).t()
        self.scale = self.bounds[1] - self.bounds[0]
        self.l_bounds = self.bounds[0]
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self._initialize_model(num_init_samples)
        self._update_current_best()
        self._maximize_kg()
        self.retrain_gp = retrain_gp
        self.num_samples = num_init_samples

    def _maximize_kg(self):
        """
        maximizes the KG acquisition function and stores the resulting value and the candidate
        :return: None
        """
        # acq_func = qKnowledgeGradient(model=self.model, current_value=self.current_best_val)
        # acq_func = qExpectedImprovement(model=self.model, best_f=self.current_best_val)
        acq_func = ExpectedImprovement(model=self.model, best_f=self.current_best_val)
        self.next_candidate, self.kg_value = optimize_acqf(acq_func, Tensor([[0], [1]]).repeat(1, self.dim),
                                                           q=1, num_restarts=self.num_restarts,
                                                           raw_samples=self.raw_samples)

    def _update_current_best(self):
        """
        return the current best solution and corresponding value
        :return: None
        """
        pm = PosteriorMean(self.model)
        self.current_best_sol, self.current_best_val = optimize_acqf(pm, Tensor([[0], [1]]).repeat(1, self.dim),
                                                                     q=1, num_restarts=self.num_restarts,
                                                                     raw_samples=self.raw_samples)

    def _function_call(self, X: Tensor) -> Tensor:
        """
        Scales the solutions to the function domain and returns the function value.
        :param X: Solutions from the relative scale of [0, 1]
        :return: function value
        """
        shape = list(X.size())
        shape[-1] = 1
        X = X * self.scale.repeat(shape) + self.l_bounds.repeat(shape)
        # TODO: adjust for minimization
        return -self.function(X).unsqueeze(1)

    def _initialize_model(self, num_init_samples: int):
        """
        initialize the GP model with num_init_samples of initial samples
        :return: None
        """
        self.train_X = torch.rand((num_init_samples, self.dim))
        self.train_Y = self._function_call(self.train_X)
        likelihood = GaussianLikelihood()
        self.model = SingleTaskGP(self.train_X, self.train_Y, likelihood)
        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        fit_gpytorch_model(mll)

    def _update_model(self, new_sample: Tensor, new_observation: Tensor):
        """
        Update the GP model with the new observation(s)
        :param new_sample: sampled point
        :param new_observation: observed function value
        :return: None
        """
        self.train_X = torch.cat((self.train_X, new_sample), 0)
        self.train_Y = torch.cat((self.train_Y, new_observation), 0)
        self.model = self.model.condition_on_observations(new_sample, new_observation)
        if self.retrain_gp:
            mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
            fit_gpytorch_model(mll)

    def sample_next(self):
        """
        sample the next point, i.e. the point that maximizes KG
        update the model and retrain if needed
        update the relevant values
        :return:
        """
        Y = self._function_call(self.next_candidate)
        self._update_model(self.next_candidate, Y)
        self._update_current_best()
        self._maximize_kg()
