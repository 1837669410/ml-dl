from utils import load_iris
from sklearn.preprocessing import LabelBinarizer
import numpy as np

(x_train, y_train), (x_test, y_test), feature_names, target_names = load_iris(random_state=42)

y_train = LabelBinarizer().fit_transform(y_train)
class_prob = y_train.mean(axis=0).reshape(1, -1)
print(class_prob)