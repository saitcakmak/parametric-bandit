import torch
from torch import Tensor
from typing import Tuple
from scipy.stats import norm


class DiscreteKG:
    """
    The discrete KG algorithm adopted from Frazier (2009)
    Algorithm is for maximization - minimize flag implemented
    """
    def __init__(self, M: int, error: float, mu_0: Tensor, Sigma_0: Tensor, minimize: bool = True):
        """
        Initialize the algorithm
        :param M: number of alternatives
        :param error: common variance of observation error
        :param mu_0: mean vector, tensor fo size M
        :param Sigma_0: covariance matrix, tensor of size M x M
        :param minimize: makes the algorithm work for minimization by negating mu
        """
        self.M = M
        self.error = error
        if mu_0.reshape(-1).size() != torch.Size([self.M]):
            raise ValueError('mu must be a tensor of size M.')
        if Sigma_0.size() != torch.Size([self.M, self.M]):
            raise ValueError('Sigma must be a tensor of size M x M.')
        self.mu = mu_0.reshape(-1).detach()
        if minimize:
            self.mu = - self.mu
        self.Sigma = Sigma_0.detach()
        self.minimize = minimize

    def find_maximizer(self):
        """
        Runs Algorithm 2 as described in the paper.
        :param mu: mean vector, tensor of size M
        :param Sigma: covariance matrix, tensor of size M x M
        :return: argmax KG(i) or argmin KG(i) if minimize
        """
        # algorithm loop
        v_star = -float('inf')
        x_star = None
        for i in range(self.M):
            a = self.mu
            b = self._sigma_tilde(i)
            b, index = torch.sort(b)
            a = a[index]
            # handle ties in b, sort a in increasing order if ties found
            for j in range(self.M):
                a[b == b[j]], _ = torch.sort(a[b == b[j]])
            # remove the redundant entries as described in the algorithm
            remaining = torch.ones(self.M, dtype=torch.bool)
            for j in range(self.M - 1):
                if b[j] == b[j+1]:
                    remaining[j] = 0
            a = a[remaining]
            b = b[remaining]
            c, A = self._algorithm_1(a, b)
            b = b[A-1]
            c = c[A]
            v = torch.log(torch.sum((b[1:] - b[:-1]) * self._f(torch.abs(c[:-1]))))
            if i == 0 or v > v_star:
                v_star = v
                x_star = i
        return x_star

    def _sigma_tilde(self, index: int) -> Tensor:
        """
        Computes sigma_tilde w.r.t the given selection
        :return: sigma_tilde, tensor of size M
        """
        sigma_tilde = self.Sigma[:, index] / torch.sqrt(self.error + self.Sigma[index, index])
        return sigma_tilde

    def _f(self, z: Tensor) -> Tensor:
        """
        The function f defined in the paper as f(z) = phi(z) + z Phi(z) where phi and Phi are normal PDF and CDF
        :param z: a Tensor of input values
        :return: corresponding f(z) values
        """
        return torch.tensor(norm.pdf(z)) + z * torch.tensor(norm.cdf(z))

    @staticmethod
    def _algorithm_1(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Algorithm 1 from Frazier 2009
        :param a: Input a
        :param b: Input b, in strictly increasing order
        :return: c and A, indices starting with 1 as in original algorithm description!
        """
        # The indices of a and b start with 0, however the rest starts with 1. Be careful about this!
        M = a.size(-1)
        c = torch.empty(M + 1)
        c[0] = -float('inf')
        c[1] = float('inf')
        A = [1]
        for i in range(1, M):
            c[i+1] = float('inf')
            done = False
            while not done:
                j = A[-1]
                c[j] = (a[j-1] - a[i]) / (b[i] - b[j-1])
                if len(A) != 1 and c[j] <= c[A[-2]]:
                    A = A[:-1]
                else:
                    done = True
            A.append(i+1)
        return c, torch.tensor(A, dtype=torch.long)

    def sample_next(self):
        raise NotImplementedError

    def update_model(self):
        raise NotImplementedError


if __name__ == '__main__':
    kg = DiscreteKG(3, 1, torch.tensor([0.0, 5, 100]), torch.tensor([[1000.0, 5, 5], [5, 10, 5], [5, 5, 10]]))
    print(kg.find_maximizer())
