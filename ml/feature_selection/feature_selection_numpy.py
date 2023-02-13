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


class Chi2Select:
    # 卡方检验
    def __init__(self, threshold):
        self.threshold = threshold

    def chi2(self, x, y):
        y = self.LabelBinarizer(y)  # [None, n_classes]
        observed = np.dot(y.T, x)  # [n_classes, None] @ [None, n_features] -> [n_classes, n_features]
        feature_count = x.sum(axis=0).reshape(1, -1)  # 每个特征的值的总和[1, n_features]
        class_prob = y.mean(axis=0).reshape(1, -1)  # 每种标签出现的概率[1, n_classes]
        expected = np.dot(class_prob.T, feature_count)  # [n_classes, 1] @ [1, n_features] -> [n_classes, n_features]
        return observed, expected

    def LabelBinarizer(self, y):
        # 标签独热编码
        label, _ = np.unique(y, return_counts=True)  # [None, ]
        label_zeros = np.zeros(shape=[y.shape[0], len(label)])  # [None, n_classes]
        for l in label:
            index = y == l
            label_zeros[index, l] = 1
        return label_zeros

    def fit(self, x, y):
        observed, expected = self.chi2(x, y)
        chisq = (observed - expected) ** 2 / expected
        print(chisq.sum(axis=0))


(x_train, y_train), (x_test, y_test), feature_names, target_names = load_iris(random_state=42)

# 方差过滤
# vs = VarianceSelect(threshold=2)
# vs.fit(x_train)
# x_train = vs.transform(x_train)

# 卡方检验
chi2 = Chi2Select(threshold=100)
chi2.fit(x_train, y_train)
