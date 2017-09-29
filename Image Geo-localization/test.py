import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    w = tf.Variable([5])
    b = tf.Variable([1])
    x = tf.placeholder(tf.int32, [3])
    y = tf.multiply(x, w) + b
    init = tf.global_variables_initializer()
    sess.run(init)
    t = sess.run(y, feed_dict={x: [5,5,5]})
    print(type(t))