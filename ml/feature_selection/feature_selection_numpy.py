import numpy as np
from utils import load_iris

class VarianceSelect:
    # 方差选择器
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, x):
        self.variance = np.var(x, axis=0)

    def transform(self, x):
        threshold_index = self.variance >= self.threshold
        return x[:, threshold_index]

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

(x_train, y_train), (x_test, y_test), feature_names, target_names = load_iris(random_state=42)
vs = VarianceSelect(threshold=2)
vs.fit(x_train)
x_train = vs.transform(x_train)