from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from utils import load_iris

(x_train, y_train), (x_test, y_test), feature_names, target_names = load_iris(random_state=42)

# 标准化
# standard = StandardScaler()
# x_train = standard.fit_transform(x_train)

# 方差选择
# vs = VarianceThreshold(threshold=2)
# vs.fit(x_train)
# x_train = vs.transform(x_train)

# 卡方检验
# c2 = SelectKBest(chi2, k=2)
# x_train = c2.fit_transform(x_train, y_train)

# F检验'
# f = SelectKBest(f_classif, k=2)
# x_train = f.fit_transform(x_train, y_train)

# 互信息法
# mic = SelectKBest(mutual_info_classif, k=2)
# x_train = mic.fit_transform(x_train, y_train)