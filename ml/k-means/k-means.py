from utils import load_iris, cal_acc
from sklearn.cluster import KMeans

(x_train, y_train), (x_test, y_test), _, _ = load_iris()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
model = KMeans(n_clusters=3)
model.fit(x_train)
print("聚类标签：", model.labels_)
print("真实标签：", y_train)
print("计算训练集聚类标签和真实标签的acc：", cal_acc(y_train, model.labels_))
print("计算测试集聚类标签和真实标签的acc：", cal_acc(y_test, model.predict(x_test)))
