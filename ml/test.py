import numpy as np

index = 9

x = np.array([5.56, 5.70, 5.91, 6.40, 6.80, 7.05, 8.9, 8.7, 9, 9.05], dtype=float)
print(np.sum(x) / len(x))
print(x - np.sum(x) / len(x))