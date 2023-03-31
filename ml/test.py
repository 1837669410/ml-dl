import numpy as np
from sklearn.metrics import mutual_info_score

x = np.array([1,2,3,4,5])
y = np.array([2,3,4,5,6])
mutual_info = mutual_info_score(x, y)
print(mutual_info)