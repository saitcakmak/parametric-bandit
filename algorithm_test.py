from botorch.test_functions import Ackley, Hartmann
from arm import ParametricArm
import torch
from torch import Tensor
from botorch.acquisition import qKnowledgeGradient, PosteriorMean
from botorch.optim import optimize_acqf
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from typing import Union
from generic_algorithm import GenericAlgorithm

# test function
ack = Ackley()
hart = Hartmann()


# construct functions g and h
def g(X: Tensor):
    return 1 - torch.exp(-X - 0.01)


def h(X: Tensor):
    return torch.exp(-X)


# construct the arm
alg = GenericAlgorithm([ack, hart], h, g, 0.5)

for i in range(2000):
    alg.sample_next()
