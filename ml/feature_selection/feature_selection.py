from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from utils import load_iris

(x_train, y_train), (x_test, y_test), feature_names, target_names = load_iris(random_state=42)

# 方差选择
# vs = VarianceThreshold(threshold=2)
# vs.fit(x_train)
# x_train = vs.transform(x_train)

# 卡方检验
# c2 = SelectKBest(chi2, k=3)
# x_train = c2.fit_transform(x_train, y_train)
