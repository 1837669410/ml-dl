from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import make_classification
import numpy as np

def load_iris(random_state=None):
    iris = datasets.load_iris()
    x = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_state)
    return (x_train, y_train), (x_test, y_test), feature_names, target_names

def load_wine(random_state=None):
    wine = datasets.load_wine()
    x = wine.data
    y = wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_state)
    return (x_train, y_train), (x_test, y_test), feature_names, target_names

def load_california_housing(random_state=None):
    home = datasets.fetch_california_housing()
    x = home.data
    y = home.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_state)
    return (x_train, y_train), (x_test, y_test)

def load_two_classification(n, random_state=None):
    x, y = make_classification(n_samples=n,
                        n_features=100,
                        n_classes=2, shift=0.1, scale=1.1, random_state=random_state)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=random_state)
    return (x_train, y_train), (x_test, y_test)

def cal_acc(y_true, y_pred):
    if len(y_true.shape) != 1 and len(y_pred.shape) != 1:
        raise ValueError("y_true和y_pred都是一维向量")
    if type(y_true) != np.ndarray:
        raise TypeError("y_true{}".format("类型错误"))
    if type(y_pred) != np.ndarray:
        raise TypeError("y_pred{}".format("类型错误"))
    return np.sum(y_true == y_pred) / len(y_true)

def cal_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def cal_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)