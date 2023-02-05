import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import load_iris

(x_train, y_train), (x_test, y_test), _, target_names = load_iris(random_state=42)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
pca = PCA(n_components=2)
pca.fit(x_train)
x_train = pca.transform(x_train)
print(sum(pca.explained_variance_ratio_))

# 可视化
plt.figure()
plt.scatter(x_train[y_train==0,0],x_train[y_train==0,1],label=target_names[0])
plt.scatter(x_train[y_train==1,0],x_train[y_train==1,1],label=target_names[1])
plt.scatter(x_train[y_train==2,0],x_train[y_train==2,1],label=target_names[2])
plt.legend()
plt.show()
