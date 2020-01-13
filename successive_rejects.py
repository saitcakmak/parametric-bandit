import torch


class SuccessiveRejects:
    """
    The successive rejects algorithm for bandit optimization
    """
    def __init__(self, K: int, N: int):
        """
        Initialize the algorithm
        :param K: Number of alternatives
        :param N: Total budget
        """
        A = torch.arange(0, K, 1)
        log_K = 0.5 + sum([1/i for i in range(2, K+1)])
        n = torch.zeros(K+1)
        n[1:] = torch.ceil(1/log_K * (N - K) / (K - A + 0.0))
        self.n = n  # budget up to given iteration
        self.K = K
        self.N = N
        self.iteration = 0

    def get_current_budget(self):
        """
        Return the budget for the current iteration
        :return: Budget per arm for the remaining arms
        """
        if self.iteration == K-1:
            raise ValueError("Budget is exhausted.")
        budget = self.n[self.iteration + 1] - self.n[self.iteration]
        self.iteration += 1
        return budget

