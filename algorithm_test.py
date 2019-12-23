import torch
from botorch.test_functions import Ackley, Hartmann
from torch import Tensor

from generic_algorithm import GenericAlgorithm

# test function - just add anything from botorch.test_functions
ack = Ackley()
hart = Hartmann()


# construct functions g and h
# play around here, these are quite arbitrary
def g(X: Tensor):
    X = torch.clamp(X, max=1)
    return X.pow(1/2)


def h(X: Tensor):
    return torch.clamp(X.pow(-1), min=-10**6, max=10**6)


# construct the arms and the algorithm
alg = GenericAlgorithm(arm_functions=[ack, ack, hart, hart], h=h, g=g, w_star0=0.5, retrain_gp=True, verbose=True)

# run the algorithm
for i in range(2000):
    alg.sample_next()
