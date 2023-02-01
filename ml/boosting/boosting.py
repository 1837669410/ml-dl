# paper: https://arxiv.org/pdf/1603.02754v3.pdf
# doc: https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.sklearn
import xgboost as xgb
import time
from utils import load_iris, cal_acc, load_california_housing, cal_r2, cal_mse

# 分类
(x_train, y_train), (x_test, y_test), _, _ = load_iris()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
model = xgb.XGBClassifier()
model.fit(x_train, y_train)
print("train", cal_acc(y_train, model.predict(x_train)))
print("test", cal_acc(y_test, model.predict(x_test)))

# 回归
(x_train, y_train), (x_test, y_test) = load_california_housing()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
model = xgb.XGBRegressor()
model.fit(x_train, y_train)
print("train: r2:{:.3f} | mse:{:.3f}".format(cal_r2(y_train, model.predict(x_train)), cal_mse(y_train, model.predict(x_train))))
print("test: r2:{:.3f} | mse:{:.3f}".format(cal_r2(y_test, model.predict(x_test)), cal_mse(y_test, model.predict(x_test))))