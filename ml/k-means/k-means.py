import matplotlib.pyplot as plt
from utils import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


(x_train, y_train), (x_test, y_test), _, target_names = load_iris()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
model = KMeans(n_clusters=3)
model.fit(x_train)
print("聚类标签：", model.labels_)

# 可视化
pca = PCA(n_components=2)
pca.fit(x_train)
x_train = pca.transform(x_train)

plt.figure()
plt.scatter(x_train[y_train==0,0],x_train[y_train==0,1],label=target_names[0])
plt.scatter(x_train[y_train==1,0],x_train[y_train==1,1],label=target_names[1])
plt.scatter(x_train[y_train==2,0],x_train[y_train==2,1],label=target_names[2])
plt.legend()
plt.show()
