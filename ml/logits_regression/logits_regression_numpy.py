import numpy as np
from utils import load_two_classification, cal_acc

class LogitsRegression():

    def __init__(self, epoch=500, lr=0.001, c=1, penalty="l2"):
        super(LogitsRegression, self).__init__()
        self.w = None
        self.b = None
        self.penalty = penalty
        self.epoch = epoch
        self.lr = lr
        self.c = c

    def cal_wb(self, x, y):
        # B_x [None 5] B_y [None, ] weight [5, ]
        # B_x @ weight [None, ]
        # B_x @ weight - B_y [None, ]
        # B_x.T @ B_x @ weight - B_y [5, ]
        weight = np.ones(shape=[x.shape[1], ])
        for _ in range(self.epoch):
            point = np.random.randint(0, len(x), size=x.shape[0] // 6)
            B_x = x[point, :]
            B_y = y[point]
            if self.penalty == "l2":
                weight = weight - self.lr * (1 / B_x.shape[0]) * B_x.T.dot(self.sigmoid(B_x.dot(weight)) - B_y) - self.c * (1 / B_x.shape[0]) * weight
        self.b, self.w = weight[0], weight[1:]

    def sigmoid(self, pred):
        return 1 / (1 + np.exp(-pred))

    def fit(self, x, y):
        x = np.concatenate((np.ones(shape=[x.shape[0],1]), x), axis=1)
        self.cal_wb(x, y)

    def score(self, y_true, y_pred):
        return "{:.2f}".format(cal_acc(y_true, y_pred))

    def predict(self, x):
        # x [None 4] w [4, ] b [1, ]
        pred = x.dot(self.w) + self.b
        pred = self.sigmoid(pred)
        index_1 = pred > 0.5
        index_mean = pred == 0.5
        index_0 = pred < 0.5
        pred[index_1] = 1
        if np.sum(index_mean) > 0:
            pred[index_mean] = np.random.choice([0, 1], size=1)
        pred[index_0] = 0
        return np.array(pred, dtype=int)

(x_train, y_train), (x_test, y_test) = load_two_classification(200)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
model = LogitsRegression()
model.fit(x_train, y_train)
print("train: ", model.score(y_train, model.predict(x_train)))
print("test: ", model.score(y_test, model.predict(x_test)))