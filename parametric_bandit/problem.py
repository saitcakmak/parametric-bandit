import torch
from gpytorch import ExactMarginalLogLikelihood
from torch import Tensor
from botorch.test_functions import SyntheticTestFunction
from parametric_bandit.test_functions.standardized_function import StandardizedFunction
from typing import List, Union, Tuple
from botorch.models import FixedNoiseGP, SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.models.transforms import Standardize
from botorch.generation import MaxPosteriorSampling


class Problem:
    """
    This is a class to hold the problem parameters
    """

    def __init__(
        self,
        functions: List[Union[SyntheticTestFunction, StandardizedFunction]],
        alternative_points: List[Tensor],
        noise_std: float = None,
        gp_update_freq: int = 10,
    ) -> None:
        """
        Initialize
        :param functions: The initialized StandardizedFunction objects
        :param alternative_points: The points where the alternatives are.
            A list of `num_alternatives x d_arm`-dim tensors.
        :param gp_update_freq: We refit the GP model every this many samples.
        """
        self.arm_count = len(functions)
        self.functions = functions
        if len(alternative_points) != self.arm_count:
            raise ValueError("Size mismatch between alternative points and arms.")
        self.alternative_points = alternative_points
        self.noise_std = noise_std
        self.gp_update_freq = gp_update_freq
        self.models: List[SingleTaskGP] = [None] * self.arm_count
        self.sample_mean = list()
        self.sample_var = list()
        self.observations = list()

    def initialize_arms(self, num_samples: Union[int, Tensor, List[Tensor]]) -> None:
        """
        Draw the initial samples and initialize the GP for each arm.

        Args:
            num_samples: If `int`, each alternative of each arm is sampled
                `num_samples` times. If `Tensor`, each alternative of `i^th` arm is
                sampled `num_samples[i]` times. If `List[Tensor]`, the `j^th`
                alternative of `i^th` arm is sampled `num_samples[i][j]` times.
        """
        if (
            self.observations != list()
            or self.sample_mean != list()
            or self.sample_var != list()
        ):
            raise RuntimeError(
                "This is only for initialization. "
                "self.values / sample_mean / sample_var need to be empty lists."
            )
        if isinstance(num_samples, int):
            for i in range(self.arm_count):
                self.observations.append(list())
                for j in range(len(self.alternative_points[i])):
                    self.observations[i].append(list())
                    point = self.alternative_points[i][j]
                    for k in range(num_samples):
                        self.observations[i][j].append(self.functions[i](point))

        elif isinstance(num_samples, Tensor):
            for i in range(self.arm_count):
                self.observations.append(list())
                for j in range(len(self.alternative_points[i])):
                    self.observations[i].append(list())
                    point = self.alternative_points[i][j]
                    for k in range(int(num_samples[i])):
                        self.observations[i][j].append(self.functions[i](point))

        elif isinstance(num_samples, list):
            for i in range(self.arm_count):
                self.observations.append(list())
                for j in range(len(self.alternative_points[i])):
                    self.observations[i].append(list())
                    point = self.alternative_points[i][j]
                    for k in range(int(num_samples[i][j])):
                        self.observations[i][j].append(self.functions[i](point))

        else:
            raise ValueError("Num_samples is not one of the appropriate types.")

        self.update_sample_stats()
        for i in range(self.arm_count):
            self.fit_gp_model(i, update=True)

    def fit_gp_model(
        self, arm: int, alternative: int = None, update: bool = False
    ) -> None:
        """
        Fits a GP model to the given arm
        :param arm: Arm index
        :param alternative: Last sampled arm alternative. Used when adding samples
            without refitting
        :param update: Forces GP to be fitted. Otherwise, it is fitted every
            self.gp_update_freq samples.
        :return: None
        """
        arm_sample_count = sum([len(e) for e in self.observations[arm]])
        if update or arm_sample_count % self.gp_update_freq == 0:
            train_X_list = list()
            train_Y_list = list()
            for j in range(len(self.alternative_points[arm])):
                for k in range(len(self.observations[arm][j])):
                    train_X_list.append(self.alternative_points[arm][j].unsqueeze(-2))
                    train_Y_list.append(self.observations[arm][j][k].unsqueeze(-2))
            train_X = torch.cat(train_X_list, dim=0)
            train_Y = torch.cat(train_Y_list, dim=0)
            if self.noise_std is None:
                model = SingleTaskGP(
                    train_X, train_Y, outcome_transform=Standardize(m=1)
                )
            else:
                model = FixedNoiseGP(
                    train_X,
                    train_Y,
                    train_Yvar=torch.tensor([self.noise_std ** 2]).expand_as(train_Y),
                    outcome_transform=Standardize(m=1),
                )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
            self.models[arm] = model
        else:
            last_point = self.alternative_points[arm][alternative].reshape(1, -1)
            last_observation = self.observations[arm][alternative][-1].reshape(1, -1)
            self.models[arm].condition_on_observations(
                last_point, last_observation, noise=self.noise_std ** 2
            )

    def update_sample_stats(self) -> None:
        """
        Updates the sample_mean and sample_var for all arms
        """
        self.sample_mean = list()
        self.sample_var = list()
        for i in range(self.arm_count):
            self.sample_mean.append(list())
            self.sample_var.append(list())
            for j in range(len(self.alternative_points[i])):
                values = torch.tensor(self.observations[i][j])
                self.sample_mean[i].append(torch.mean(values))
                self.sample_var[i].append(torch.var(values))

    def new_sample(self, arm: int, alternative: int, ocba: bool) -> None:
        """
        Sample from a given alternative and update its model / stats
        :param arm: arm index
        :param alternative: alternative index
        :param ocba: If OCBA, we do not update the GP model.
            If not, we don't update sample stats. Saves on computation.
        """
        self.observations[arm][alternative].append(
            self.functions[arm](self.alternative_points[arm][alternative])
        )
        if ocba:
            values = torch.tensor(self.observations[arm][alternative])
            self.sample_mean[arm][alternative] = torch.mean(values)
            self.sample_var[arm][alternative] = torch.var(values)
        else:
            self.fit_gp_model(arm, alternative)

    def get_arm_gp_mean_cov(self, arm: int = None) -> Tuple[Tensor, Tensor]:
        """
        Get the mean and covariance matrix of a given arm or all arms.
        :param arm: If given, return only for the arm. Otherwise, return all.
        :return: Mean and Covariance matrix
        """
        if arm is None:
            alternative_count = list()
            for i in range(self.arm_count):
                alternative_count.append(len(self.alternative_points[i]))
            total_size = sum(alternative_count)
            mean = torch.empty(total_size)
            covariance = torch.zeros((total_size, total_size))
            for i in range(self.arm_count):
                post = self.models[i].posterior(self.alternative_points[i])
                left_index = sum(alternative_count[:i])
                right_index = sum(alternative_count[: i + 1])
                mean[left_index:right_index] = post.mean.squeeze()
                covariance[
                    left_index:right_index, left_index:right_index
                ] = post.mvn.covariance_matrix
        else:
            post = self.models[arm].posterior(self.alternative_points[arm])
            mean = post.mean
            covariance = post.mvn.covariance_matrix
        return mean, covariance

    def get_arm_stats(self, arm: int = None) -> Tuple[Tensor, Tensor]:
        """
        Returns Tensors sample mean and variance for the given arm.
        Returns a flattened Tensors of sample mean and variance if no arm specified.
        :param arm: If given, return only for the arm. Otherwise, return all.
        :return: Sample mean and variance as described
        """
        if arm is None:
            sample_mean_list = list()
            sample_var_list = list()
            for i in range(self.arm_count):
                sample_mean_list.append(torch.tensor(self.sample_mean[i]).reshape(-1))
                sample_var_list.append(torch.tensor(self.sample_var[i]).reshape(-1))
            sample_mean = torch.cat(sample_mean_list, dim=0)
            sample_var = torch.cat(sample_var_list, dim=0)
        else:
            sample_mean = torch.tensor(self.sample_mean[arm]).reshape(-1)
            sample_var = torch.tensor(self.sample_var[arm]).reshape(-1)
        return sample_mean, sample_var

    def get_true_val(self) -> List[Tensor]:
        """
        The true performance of the arms
        :return: the true function value
        """
        true_val = list()
        for i in range(self.arm_count):
            true_val.append(torch.empty(len(self.alternative_points[i])))
            for j in range(len(self.alternative_points[i])):
                arm_val = self.functions[i].evaluate_true(
                    self.alternative_points[i][j].unsqueeze(0)
                )
                true_val[i][j] = arm_val
        return true_val

    def get_outer_stats(
        self, arm: int = None, num_samples: int = 100
    ) -> Tuple[Tensor, Tensor]:
        """
        This returns the E[max] and Std[max] for the given arm estimated
        using samples from the GP. If no arm is specified, it loops over
        all arms and returns Tensors of appropriate size.
        :param arm: Arm index to return stats for, returns for all arms if None.
        :param num_samples: Number of samples from GP to use
        :return: Estimates of E[max] and Std[max]
        """
        if arm is None:
            mean = torch.empty(self.arm_count)
            std = torch.empty(self.arm_count)
            for i in range(self.arm_count):
                mean[i], std[i] = self.get_outer_stats(arm=i, num_samples=num_samples)
            return mean, std
        else:
            model = self.models[arm]
            points = self.alternative_points[arm]
            post = model.posterior(points)
            samples = post.rsample(torch.Size([num_samples]))
            max_sample, _ = torch.max(samples, dim=1)
            mean = torch.mean(max_sample)
            std = torch.std(max_sample)
            return mean, std

    def mps_sample(self, arm: int) -> Tensor:
        """
        Uses MaxPosteriorSampling and returns the maximizer of a GP sample path.

        :param arm: The arm to sample from
        :return: max sample point
        """
        model = self.models[arm]
        dim = self.functions[arm].dim
        mps = MaxPosteriorSampling(model)
        X = self.alternative_points[arm].reshape(1, -1, dim)
        TS_sample = mps(X)
        return TS_sample

    def ts_sample(self, arm: int) -> Tuple[int, Tensor]:
        r"""
        Draw the next Thompson sample from the given arm. The difference between this
        and MPS is that MPS doesn't return the value of the sample.

        Args:
            arm: The arm to sample from

        Returns:
            The max sample point and the sample value
        """
        model = self.models[arm]
        X = self.alternative_points[arm]
        sample_path = model.posterior(X).rsample()
        max_val, max_ind = torch.max(sample_path, dim=-2)
        return int(max_ind), max_val.squeeze()

    def ts_all_arms(self) -> Tuple[int, int, Tensor]:
        r"""
        Runs ts_sample for each arm and returns the maximizer.

        Returns:
            The arm with the best sample, the alternative, and the value
        """
        best_arm = None
        best_alternative = None
        best_value = torch.tensor([-float("Inf")])
        for i in range(self.arm_count):
            alternative, value = self.ts_sample(i)
            if value > best_value:
                best_arm = i
                best_alternative = alternative
                best_value = value
        return best_arm, best_alternative, best_value
