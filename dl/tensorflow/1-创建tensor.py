import tensorflow as tf
import numpy as np

# 1: list
a = [1,2,3]
print("list:{}".format(a))
# 2: array
b = np.array([1,2,3])
print("array:{}".format(b))
# 3: tensor
c = tf.constant([1,2,3])
print(c)
print("tensor:{}".format(c))
# 4: tensor to numpy
d = c.numpy()
print("array:{}".format(d))