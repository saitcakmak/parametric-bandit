import torch
from torch import Tensor


class OCBA:
    """
    The fully sequential OCBA algorithm. The algorithm description is taken from Wu & Zhou 2018
    Algorithm is for minimization
    """

    def __init__(self, K: int, N: int, N_0: int, randomized: bool = False):
        """
        Initialize the algorithm
        :param K: number of alternatives
        :param N: total budget
        :param N_0: initial budget
        :param randomized: if True OCBA-R is used, else OCBA-D is used.
        """
        self.K = K
        self.N = N
        self.remaining = N - K * N_0
        self.randomized = randomized
        self.total_sampled = torch.ones(K) * N_0  # total samples allocated to each arm so far

    def next_sample(self, x_bar: Tensor, s_bar: Tensor):
        """
        Runs the OCBA loop to suggest the next sample
        :param x_bar: performance estimate of the given arm
        :param s_bar: standard deviation / uncertainty estimate of the given arm
        :return: Next one to sample
        """
        if x_bar.reshape(-1).size() != torch.Size([self.K]):
            raise ValueError('x_bar must be a Tensor of size K.')
        if s_bar.reshape(-1).size() != torch.Size([self.K]):
            raise ValueError('s_bar must be a Tensor of size K.')
        if self.remaining == 0:
            raise ValueError('Budget already exhausted.')
        if any(s_bar <= 0):
            raise ValueError('s_bar must be strictly positive.')
        x_bar = x_bar.reshape(-1)
        s_bar = s_bar.reshape(-1)
        # calculate alpha values
        x_best, best_index = torch.min(x_bar, dim=-1)
        beta = torch.zeros(self.K)
        # define delta with a tiny perturbation to avoid division by zero
        delta = x_bar - x_best + 10 ** -8
        beta = s_bar.pow(2) / delta.pow(2)
        # beta[best_index] is not included in the summation below, thus set to 0
        beta[best_index] = 0
        beta[best_index] = s_bar[best_index] * torch.sqrt(torch.sum(beta / s_bar.pow(2)))
        alpha = beta / torch.sum(beta)
        if not self.randomized:
            i_star = torch.argmax(alpha / self.total_sampled)
        else:
            uniform = torch.rand(1)
            sum = 0
            for i in range(self.K):
                sum += alpha[i]
                if uniform < sum:
                    i_star = i
                    break
        self.remaining -= 1
        self.total_sampled[i_star] += 1
        return i_star


if __name__ == '__main__':
    ocba = OCBA(3, 50, 3, False)
    x_b = torch.tensor([0, 20, 10])
    s_b = torch.tensor([3, 5, 3])
    print(ocba.next_sample(x_b, s_b))
