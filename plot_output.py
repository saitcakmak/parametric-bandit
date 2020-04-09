"""
This will read the output of run_exp and plot the average regret
"""
import torch
import matplotlib.pyplot as plt


filename = "output/sparse_6x50_std50.pt"
data = torch.load(filename)
results = data['results']

keys = list(results.keys())
keys.sort()
key_count = len(keys)

ocba_regret = torch.empty(key_count)
ocba_std = torch.empty(key_count)
kg_regret = torch.empty(key_count)
kg_std = torch.empty(key_count)
composite_regret = torch.empty(key_count)
composite_std = torch.empty(key_count)
for i in range(key_count):
    ocba_regret[i] = torch.mean(results[keys[i]][:, 0])
    ocba_std[i] = torch.std(results[keys[i]][:, 0])
    kg_regret[i] = torch.mean(results[keys[i]][:, 1])
    kg_std[i] = torch.std(results[keys[i]][:, 1])
    composite_regret[i] = torch.mean(results[keys[i]][:, 2])
    composite_std[i] = torch.std(results[keys[i]][:, 2])

alpha = 0.3
plt.plot(keys, ocba_regret, label="OCBA")
plt.fill_between(keys, ocba_regret-ocba_std, ocba_regret+ocba_std, alpha=alpha)
plt.plot(keys, kg_regret, label="KG")
plt.fill_between(keys, kg_regret-kg_std, kg_regret+kg_std, alpha=alpha)
plt.plot(keys, composite_regret, label="Composite")
plt.fill_between(keys, composite_regret-composite_std, composite_regret+composite_std, alpha=alpha)

plt.legend()
plt.grid(True)
plt.xlabel("Budget")
plt.ylabel("Regret")
plt.title(filename)
# plt.yscale("log")
print(results[keys[0]].size(0))
plt.show()
