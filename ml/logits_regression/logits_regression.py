from sklearn.linear_model import LogisticRegression
from utils import load_two_classification, cal_acc

(x_train, y_train), (x_test, y_test) = load_two_classification(200)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
model = LogisticRegression()
model.fit(x_train, y_train)
print("train: ", cal_acc(y_train, model.predict(x_train)))
print("test: ", cal_acc(y_test, model.predict(x_test)))