import numpy as np
from utils import load_california_housing, cal_r2, cal_mse

class LinearRegression():

    def __init__(self):
        self.w = None
        self.b = None

    def cal_wb(self, x, y):
        # w = inv(X.T @ X) @ X.T @ Y
        # X [None 9] X.T [9 None] Y[None, ]
        # X.T @ X [9 9]
        # inv(X.T @ X) [9 9]
        # inv(X.T @ X) @ X.T [9 None]
        # inv(X.T @ X) @ X.T @ Y [9, ]
        temp = x.T.dot(x)
        temp = np.linalg.inv(temp).dot(x.T).dot(y)
        self.b, self.w = temp[0], temp[1:]

    def fit(self, x, y):
        # 拼接上一列1，以此获得截距b
        x = np.concatenate((np.ones(shape=[x.shape[0],1]), x), axis=1)
        self.cal_wb(x, y)

    def predict(self, x):
        # x [None 8] w [8, ] b [1, ]
        # x @ w [None, ]
        # broadcast x @ w + b [None, ]
        return x.dot(self.w) + self.b

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_california_housing()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    model = LinearRegression()
    model.fit(x_train, y_train)
    print("train: r2:{:.3f} | mse:{:.3f}".format(cal_r2(y_train, model.predict(x_train)), cal_mse(y_train, model.predict(x_train))))
    print("test: r2:{:.3f} | mse:{:.3f}".format(cal_r2(y_test, model.predict(x_test)), cal_mse(y_test, model.predict(x_test))))