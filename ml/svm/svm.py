from sklearn.svm import SVC
from utils import load_iris

(x_train, y_train), (x_test, y_test), _, _ = load_iris()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
model = SVC(kernel="rbf")
model.fit(x_train, y_train)
print("train", model.score(x_train, y_train))
print("test", model.score(x_test, y_test))