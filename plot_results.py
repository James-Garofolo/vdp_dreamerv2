import matplotlib.pyplot as plt
import numpy as np

baseline = np.loadtxt("avgs_baseline.csv")
exp = np.loadtxt("avgs_exp.csv")
t = np.arange(exp.shape[0])
idx = exp.shape[0]-baseline.shape[0]

plt.plot(t[idx:], baseline, label="Dreamerv2")
plt.plot(t, exp, label="Uncertainty Aware")
plt.legend()
plt.ylabel("average score")
plt.show()