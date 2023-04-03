import numpy as np
import matplotlib.pyplot as plt
from utils import load_iris
from sklearn.decomposition import PCA

class KMeans():

    def __init__(self, n, epoch=500):
        self.n = n   # 聚类类别数
        self.n_center = {}   # 每个类别的中心点
        self.sample_label = []   # 储存每个样本的类别
        self.epoch = epoch   # 循环迭代次数

    def _get_init(self):
        # 初始化中心点
        n_sample = self.x.shape[0] // self.n   # 得到每类数据的样本量，为了方便处理直接整除
        self.n_center = {}                     # 保存每一个类别的中心点
        for i in range(self.n):
            choose_point = np.random.randint(0, self.x.shape[0], size=n_sample)   # 样本的选择点
            choose_sample = self.x[choose_point,:]                                # 选取的样本
            self.n_center[i] = np.mean(choose_sample, axis=0)                     # 初始中心点

    def cla_distance(self, center):
        # 计算每个样本到中心点的距离
        sample_distance = {}   # 样本对所有中心点的距离
        for i in range(len(self.x)):
            temp = []   # 暂存每个类别的距离
            for j in range(self.n):
                temp.append(np.sum((self.x[i,:] - center[j]) ** 2))
            sample_distance[i] = temp
        return sample_distance

    def cla_label(self, sample_distance):
        for i, v in sample_distance.items():
            self.sample_label.append(np.argmin(v))

    def cla_center(self):
        for i in range(self.n):
            temp = self.x[np.array(self.sample_label) == i, :]   # 暂存该类别的原数据
            self.n_center[i] = np.mean(temp, axis=0)   # 重新计算中心点

    def fit(self, x):
        self.x = x
        # 1 初始化中心点
        self._get_init()
        # 2 训练
        for i in range(self.epoch):
            # 清空类别属性
            self.sample_label = []
            # 2.1 计算每个样本到中心点的距离
            sample_distance = self.cla_distance(self.n_center)
            # 2.2 得到每个样本的聚类类别
            self.cla_label(sample_distance)
            # 2.3 根据类别更新中心点
            self.cla_center()
        # 更新sample_label的格式为np.ndarray
        self.sample_label = np.array(self.sample_label)


(x_train, y_train), (x_test, y_test), _, target_names = load_iris(random_state=100)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = KMeans(n=3)
model.fit(x_train)

# 可视化
pca = PCA(n_components=2, random_state=100)
pca.fit(x_train)
x_train = pca.transform(x_train)

plt.figure()
plt.scatter(x_train[model.sample_label==0,0],x_train[model.sample_label==0,1],label=target_names[0])
plt.scatter(x_train[model.sample_label==1,0],x_train[model.sample_label==1,1],label=target_names[1])
plt.scatter(x_train[model.sample_label==2,0],x_train[model.sample_label==2,1],label=target_names[2])
plt.legend()
plt.show()