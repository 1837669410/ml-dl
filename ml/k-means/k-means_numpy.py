import numpy as np
import matplotlib.pyplot as plt
from utils import load_iris
from sklearn.decomposition import PCA

class KMeans():

    def __init__(self, n, epoch=300, init_mode="kmeans++"):
        self.n = n   # 聚类类别数
        self.n_center = {}   # 每个类别的中心点
        self.sample_label = []   # 储存每个样本的类别
        self.epoch = epoch   # 循环迭代次数
        self.clip = 1e-6    # 设置变化距离限制如果上一次变化和这一次距离变化小于该数字则直接停止训练
        self._init_mode = init_mode   # 设置初始化中心点的模式

    def _get_init(self):
        # 初始化中心点
        if self._init_mode == "random":
            n_sample = self.x.shape[0] // self.n   # 得到每类数据的样本量，为了方便处理直接整除
            self.n_center = {}                     # 保存每一个类别的中心点
            for i in range(self.n):
                choose_point = np.random.randint(0, self.x.shape[0], size=n_sample)   # 样本的选择点
                choose_sample = self.x[choose_point,:]                                # 选取的样本
                self.n_center[i] = np.mean(choose_sample, axis=0)                     # 初始中心点
            print(self.n_center)
        elif self._init_mode == "kmeans++":
            _init_choose_point = np.random.randint(0, self.x.shape[0], size=1)   # 选择最开始的一个点
            self.n_center[0] = self.x[_init_choose_point,:]   # 将最开始的点当成选定的第一个中心点
            for i in range(1, self.n):
                sample_distance = np.zeros(shape=[self.x.shape[0], ])   # 样本到现有中心点的距离总距离
                for j in list(self.n_center.keys()):
                    print(i, j)
                    sample_distance += (np.sum((self.x - self.n_center[j]) ** 2, axis=1))
                choose_prob = sample_distance / np.sum(sample_distance)
                choose_point = np.random.choice(np.arange(0, self.x.shape[0]), p=choose_prob)
                self.n_center[i] = self.x[choose_point, :]

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
        total_distance = 0
        for i, v in sample_distance.items():
            self.sample_label.append(np.argmin(v))
            total_distance += np.min(v)
        return total_distance

    def cla_center(self):
        for i in range(self.n):
            temp = self.x[np.array(self.sample_label) == i, :]   # 暂存该类别的原数据
            self.n_center[i] = np.mean(temp, axis=0)   # 重新计算中心点

    def fit(self, x):
        pre_distance = 10000
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
            distance = self.cla_label(sample_distance)
            if pre_distance - distance < self.clip:
                break
            pre_distance = distance
            # 2.3 根据类别更新中心点
            self.cla_center()
        # 更新sample_label的格式为np.ndarray
        self.sample_label = np.array(self.sample_label)


(x_train, y_train), (x_test, y_test), _, target_names = load_iris()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

model = KMeans(n=3)
model.fit(x_train)

# 可视化
pca = PCA(n_components=2)
pca.fit(x_train)
x_train = pca.transform(x_train)

plt.figure()
plt.scatter(x_train[model.sample_label==0,0],x_train[model.sample_label==0,1],label=target_names[0])
plt.scatter(x_train[model.sample_label==1,0],x_train[model.sample_label==1,1],label=target_names[1])
plt.scatter(x_train[model.sample_label==2,0],x_train[model.sample_label==2,1],label=target_names[2])
plt.legend()
plt.show()
