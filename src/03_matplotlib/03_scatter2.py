import matplotlib.pyplot as plt
import numpy as np

x = np.random.rand(10)
y = np.random.rand(10)

plt.xlim(-1, 2)
plt.ylim(-1, 2)

plt.xlabel("my x axis")
plt.ylabel("my y axis")

plt.scatter(x, y, s=100, label="my data", marker='x', color="green")
plt.legend()
plt.show()
