import tensorflow as tf
import numpy as np

with tf.Session() as sess:
    # x = tf.Variable(tf.random_normal([1, 13, 13, 256]))
    # convg3x3 = tf.nn.conv2d(x, tf.Variable(tf.random_normal([3, 3, 256, 32])), [1, 1, 1, 1], "SAME")
    # convg5x5 = tf.nn.conv2d(x, tf.Variable(tf.random_normal([5, 5, 256, 32])), [1, 1, 1, 1], "SAME")
    # convg7x7 = tf.nn.conv2d(x, tf.Variable(tf.random_normal([7, 7, 256, 20])), [1, 1, 1, 1], "SAME")
    # convg = tf.concat([convg3x3, convg5x5, convg7x7], -1)
    # convw = tf.nn.conv2d(convg, tf.Variable(tf.random_normal([1, 1, 84, 84])), [1, 1, 1, 1], "VALID")
    #
    # m = tf.Variable(tf.random_normal([24, 13, 13, 1]))
    # m_expand = tf.tile(m, [1, 1, 1, 64])
    # conva = tf.Variable(tf.random_normal([24, 13, 13, 64]))
    # conva = tf.multiply(m_expand, conva)
    # sess.run(tf.global_variables_initializer())
    # refervectors = sess.run(conva)
    # print(conva.get_shape())
    # # print(sess.run(conva))
    # print(type(refervectors))

    # conv5_reshape = tf.Variable(tf.zeros([3, 2, 2, 2]))
    # c_reshape = tf.Variable([[1, 2], [3, 4]], dtype='float32')
    # residuals = tf.subtract(conv5_reshape, c_reshape)
    # sess.run(tf.global_variables_initializer())
    # print sess.run(residuals)
    # print residuals.get_shape()

    a = tf.Variable(tf.random_normal([5, 13, 13, 64]))
    inputs = tf.Variable(tf.random_normal([5, 13, 13, 1]))
    m = tf.tile(inputs, [1, 1, 1, 64])
    # m = tf.multiply(a, inputs)
    sess.run(tf.global_variables_initializer())
    print m.get_shape()