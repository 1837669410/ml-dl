import numpy as np
import matplotlib.pyplot as plt
from utils import load_iris

class PCA:

    def __init__(self, n):
        self.n = n

    def fit(self, x):
        # 1、标准化
        x = x - np.mean(x, axis=0)
        # 2、求cov矩阵
        cov = np.cov(x.T)
        # 3、求特征值ew，特征向量ev
        ew, ev = np.linalg.eig(cov)
        # 4、排序拿到排序的index
        e_index = np.argsort(ew)[::-1]
        self.ew, self.ev = ew[e_index], ev[:, e_index]
        # 6、拿到方差贡献率
        self.std_rate = np.sum(ew[:self.n]) / np.sum(ew)

    def transform(self, x):
        x = x - np.mean(x, axis=0)
        return np.dot(x, self.ev[:,:self.n])

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

(x_train, y_train), (x_test, y_test), _, target_names = load_iris(random_state=42)
pca = PCA(2)
pca.fit(x_train)
x_train = pca.transform(x_train)
print(x_train)
print(pca.std_rate)

# 可视化
plt.figure()
plt.scatter(x_train[y_train==0,0],x_train[y_train==0,1],label=target_names[0])
plt.scatter(x_train[y_train==1,0],x_train[y_train==1,1],label=target_names[1])
plt.scatter(x_train[y_train==2,0],x_train[y_train==2,1],label=target_names[2])
plt.legend()
plt.show()