from sklearn.linear_model import LinearRegression
from utils import load_california_housing, cal_r2, cal_mse

(x_train, y_train), (x_test, y_test) = load_california_housing(random_state=55)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
model = LinearRegression()
model.fit(x_train, y_train)
print("train: r2:{:.3f} | mse:{:.3f}".format(cal_r2(y_train, model.predict(x_train)), cal_mse(y_train, model.predict(x_train))))
print("test: r2:{:.3f} | mse:{:.3f}".format(cal_r2(y_test, model.predict(x_test)), cal_mse(y_test, model.predict(x_test))))