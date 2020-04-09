"""
run a set of experiments here
"""
import torch
from experiment import single_rep
from time import time
from torch.distributions import Bernoulli

budget_list = [10, 20, 50, 100, 200]
rep = 10
n_arm = 50
n = [n_arm] * 6
obs_std = 50.
# num_init_samples = 1
m = Bernoulli(0.2)
num_init_samples = [m.sample([n_arm]),
                    m.sample([n_arm]),
                    m.sample([n_arm]),
                    m.sample([n_arm]),
                    m.sample([n_arm]),
                    m.sample([n_arm]),
                    ]

for e in num_init_samples:
    if sum(e) == 0:
        e[0] = 1

# TODO: do it so that instead of just running it again and again, we add samples to last one
#       optimize by avoiding gp fitting with ocba runs

output_file = 'output/sparse_6x50_std50.pt'
try:
    output = torch.load(output_file)
    results = output['results']
except FileNotFoundError:
    results = dict()

start = time()
for i in range(len(budget_list)):
    if budget_list[i] not in results.keys():
        results[budget_list[i]] = torch.empty((rep, 3))
        old_results = None
    else:
        old_results = results[budget_list[i]]
        if old_results.size(0) >= rep:
            continue
        else:
            results[budget_list[i]] = torch.empty((rep, 3))
    for j in range(rep):
        if old_results is not None and j < old_results.size(0):
            results[budget_list[i]][j] = old_results[j]
            continue
        res = single_rep(seed=0,
                         n=n,
                         obs_std=obs_std,
                         N=budget_list[i],
                         num_init_samples=num_init_samples
                         )
        results[budget_list[i]][j] = torch.tensor(res)
        print('budget: %d, rep: %d, time: %s' % (budget_list[i], j, time() - start))
    output = {'results': results,
              'n': n,
              'obs_std': obs_std,
              'num_init_samples': num_init_samples
              }

    torch.save(output, output_file)
