from botorch.test_functions import Ackley, Hartmann
from arm import ParametricArm
import torch
from torch import Tensor
from botorch.acquisition import qKnowledgeGradient, PosteriorMean
from botorch.optim import optimize_acqf
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# test function
ack = Ackley()
# construct the arm
arm1 = ParametricArm(ack)

pm = PosteriorMean(arm1.model)

cand, val = optimize_acqf(
    pm, Tensor([[0], [1]]).repeat(1, 2), q=1, num_restarts=10, raw_samples=100
)

plt.figure()
ax = plt.axes(projection="3d")
k = 40  # number of points in x and y
x = torch.linspace(0, 1, k)
xx = x.view(-1, 1).repeat(1, k)
yy = x.repeat(k, 1)
xy = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2)
means = arm1.model.posterior(xy).mean
ax.scatter3D(
    xx.reshape(-1).numpy(), yy.reshape(-1).numpy(), means.detach().reshape(-1).numpy()
)
plt.show(block=False)
plt.pause(0.01)
