from sklearn.feature_selection import VarianceThreshold
from utils import load_iris

(x_train, y_train), (x_test, y_test), feature_names, target_names = load_iris(random_state=42)
vs = VarianceThreshold(threshold=2)
vs.fit(x_train)
x_train = vs.transform(x_train)