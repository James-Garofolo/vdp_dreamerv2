import numpy as np
import matplotlib.pyplot as plt
import os


files = os.listdir(r"training_curves")


#print(f.replace(".csv", "")[8:])

maxes = []
scalers = []
shortest = []
"""for f in files:
    a = np.loadtxt("training_curves/" + f, delimiter=",")
    plt.plot(a, label=f.replace(".csv", "")[8:])"""

    
for f in files:
    a = np.loadtxt("training_curves/" + f, delimiter=",")
    maxes.append(a.max())
    shortest.append(a.size)
    scalers.append(f.replace(".csv", "")[8:])

maxes = np.array(maxes)
print(np.sort(maxes))
print(np.sort(np.array(shortest)))
scalers = np.array(scalers)[np.argsort(maxes)]

for i in range(-1, -6, -1):
    print(scalers[i])
    a = np.loadtxt("training_curves/avgs_exp" + scalers[i] + ".csv", delimiter=",")
    plt.plot(a, label=scalers[i])

plt.xlabel("training episode")
plt.ylabel("average score")
plt.legend()
plt.show()