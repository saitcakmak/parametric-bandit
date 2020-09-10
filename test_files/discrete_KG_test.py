import torch
from botorch.test_functions import Branin
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.models.transforms import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from parametric_bandit.discrete_KG import DiscreteKGAlg

torch.manual_seed(0)

# generate input
n = 10
noise_std = 0.1
function = Branin(noise_std=0.1)
dim = function.dim
train_X = torch.rand((n, dim))
train_Y = function(train_X).unsqueeze(-1)

# fit model
gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_model(mll)

# get mu and Sigma
mu = gp.posterior(train_X).mean
Sigma = gp.posterior(train_X).mvn.covariance_matrix

# initiate the algorithm for testing
dkg = DiscreteKGAlg(M=n, error=noise_std ** 2, mu_0=mu, Sigma_0=Sigma)
print(dkg.find_maximizer())
