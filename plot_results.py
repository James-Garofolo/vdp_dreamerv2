import matplotlib.pyplot as plt
import numpy as np

baseline = np.loadtxt("avgs_baseline.csv")
exp = np.loadtxt("avgs_exp.csv")

print(baseline.shape, exp.shape)