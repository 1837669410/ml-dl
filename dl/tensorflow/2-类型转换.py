import tensorflow as tf

# create tensor
a = tf.constant([1,2,3], dtype=tf.int64)
print(a)
# convert to tf.int32
b = tf.cast(a, dtype=tf.int32)
print(b)
# convert to tf.float32
c = tf.cast(a, dtype=tf.float32)
print(c)