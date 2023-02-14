import numpy as np
import scipy.special as sc
from utils import load_iris

class StandardScaler:
    # 标准化
    def __init__(self):
        pass

    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        self.std = np.std(x, axis=0)

    def transform(self, x):
        return (x - self.mean) / self.std

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

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
    # 卡方检验(卡方值越大差异越大)
    def __init__(self, k):
        self.k = k

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
        chi2_score = (observed - expected) ** 2 / expected   # 计算公式：sum((观察值 - 期望值) ** 2 / 期望值)
        self.chi2_score = chi2_score.sum(axis=0)
        self.p_score = sc.chdtrc(len(observed)-1, self.chi2_score)

    def transform(self, x):
        chi2_index = np.argsort(self.chi2_score)[::-1][:self.k]
        return x[:, chi2_index]

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

(x_train, y_train), (x_test, y_test), feature_names, target_names = load_iris(random_state=42)

# 标准化
# standard = StandardScaler()
# x_train = standard.fit_transform(x_train)

# 方差过滤
# vs = VarianceSelect(threshold=2)
# vs.fit(x_train)
# x_train = vs.transform(x_train)

# 卡方检验
# chi2 = Chi2Select(k=3)
# chi2.fit(x_train, y_train)
# x_train = chi2.transform(x_train)