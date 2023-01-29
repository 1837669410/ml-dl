# 参考资料: https://blog.csdn.net/qq_45954444/article/details/108287259?spm=1001.2014.3001.5501
from sklearn.ensemble import RandomForestClassifier
from utils import load_iris

(x_train, y_train), (x_test, y_test), _, _ = load_iris()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
model = RandomForestClassifier()
model.fit(x_train, y_train)
print("train", model.score(x_train, y_train))
print("test", model.score(x_test, y_test))