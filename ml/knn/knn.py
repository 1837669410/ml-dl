from utils import load_iris
from sklearn.neighbors import KNeighborsClassifier

(x_train, y_train), (x_test, y_test), _, _ = load_iris()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
print("train", model.score(x_train, y_train))
print("test", model.score(x_test, y_test))