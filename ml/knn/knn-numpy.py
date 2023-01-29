import numpy as np
from utils import load_iris, cal_acc

class KNN():

    def __init__(self, k):
        self.k = k

    def calculate_distance(self, x):
        # result: 每个样本与其他样本之间的距离
        # prob_result: 与其最近的k个样本的index
        result = {}
        prob_result = {}
        for i, v in enumerate(x):
            # x [None 4] v [1 4]
            # 广播机制 x - v [None 4]
            distance = np.sum(np.square(x - v), axis=1)
            result[i] = distance
            # 计算prob
            k_min_distance_index = np.argsort(distance)[1:self.k+1]
            prob_result[i] = k_min_distance_index
        return result, prob_result

    def fit(self, x, y):
        self.x, self.y = x, y
        _, prob_result = self.calculate_distance(x)
        result = []
        for i, v in prob_result.items():
            pred = y[v]
            pred, pred_count = np.unique(pred, return_counts=True)
            pred_max_index = np.argsort(pred_count)[::-1]
            result.append(pred[pred_max_index][0])
        self.result = np.array(result)

    def score(self, y_true, y_pred):
        return "{:.2f}".format(cal_acc(y_true, y_pred))

    def predict(self, x):
        if x.shape[0] == self.x.shape[0] and np.sum(x == self.x) == self.x.shape[0] * self.x.shape[1]:
            return self.result
        else:
            shape = x.shape[0]
            x_new = np.concatenate((x, self.x), axis=0)
            result, _ = self.calculate_distance(x_new)
            pred = []
            for i in range(0, shape):
                temp = result[i][shape:]
                min_distance_index = np.argsort(temp)[0]
                pred.append(self.result[min_distance_index])
            return np.array(pred)

(x_train, y_train), (x_test, y_test), _, _ = load_iris()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
model = KNN(k=5)
model.fit(x_train, y_train)
print("train", model.score(y_train, model.predict(x_train)))
print("test", model.score(y_test, model.predict(x_test)))
