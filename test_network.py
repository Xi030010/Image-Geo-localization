#coding:utf-8

import tensorflow as tf
import numpy as np

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights',
                                  shape=[filter_height, filter_width,
                                         input_channels / groups, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(x, weights)

        # In the cases of multiple groups, split inputs & weights and
        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(x, groups, 3)
            weight_groups = tf.split(weights, groups, 3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(axis = 3, values = output_groups)

        # Add biases
        # bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
        bias = tf.nn.bias_add(conv, biases)

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)

        return relu


def fc(x, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu == True:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x,
             name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)

# downsample conv5 from BxWxHxD to Bx13x13xD
def downsample(conv5):

    return conv5

# upsample Reweighting Mask m from Bx13x13x1 to BxWxHx1
def upsample(convw):

    return convw

X = tf.placeholder(tf.float32, [None, 227, 227, 3])

K = 64

batch_size = 24

conv1 = conv(X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

# 2nd Layer: Conv (w ReLu) -> Lrn -> Poolwith 2 groups
conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2')
norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

# 3rd Layer: Conv (w ReLu)
conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3')

# 4th Layer: Conv (w ReLu) splitted into two groups
conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

# 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')

# get W, H, D from conv5
# D is 256
D = conv5.get_shape()[3].value
# W is 13
W = conv5.get_shape()[1].value
# H is 13
H = conv5.get_shape()[2].value

# Contextual Reweighting Network
conv5 = downsample(conv5)
# g Multiscale Context Filters, dimension is Bx13x13x84
convg3x3 = conv(conv5, 3, 3, 32, 1, 1, name='convg3x3')
convg5x5 = conv(conv5, 5, 5, 32, 1, 1, name='convg5x5')
convg7x7 = conv(conv5, 7, 7, 20, 1, 1, name='convg7x7')
convg = tf.concat([convg3x3, convg5x5, convg7x7], -1)
# w Accumulation Weight, 13x13x84 to 13x13x1
convw = conv(convg, 1, 1, 1, 1, 1, name='convw')
# Bx13x13x1 to BxWxHx1
m = upsample(convw)

# NetVLAD pooling layer, based on AlexNet

# soft_assignment
# x -> s, BxWxHxD is the dimension of conv5 output of AlexNet, and dimension of convs is BxWxHxK
k_h = 1
k_w = 1
c_o = K
s_h = 1
s_w = 1
convs = conv(conv5, k_h, k_w, c_o, s_h, s_w, padding="VALID", name="convs")
# print(convs.get_shape())
# s -> a, a is BxWxHxK
conva = tf.nn.softmax(convs)

# tile m from BxWxHx1 to BxWxHxK
m_tile = tf.tile(m, [1, 1, 1, K])
# get conva multiply m
conva_with_m = tf.multiply(conva, m_tile)

# parameter ck, totally we have k cks.The dimension of ck is KxD.
c = tf.Variable(tf.random_normal([K, D]))  # 2-D python array

# VLAD core, get vector V whose dimension is BxKxD. Let's try to use a loop to assign V firstly.

# a: reshape a from BxWxHxK to BxNxK
conva_reshape = tf.reshape(conva_with_m, shape=[-1, W * H, K])
# a: transpose a from BxNxK to BxKxN
conva_transpose = tf.transpose(conva_reshape, [0, 2, 1])

# reshape conv5 from BxWxHxD to BxNxD
conv5_reshape = tf.reshape(conv5, [-1, W * H, D])

# BxNxK and BxNxD to BxKxD
for i in range(K):
    # residual is BxNxD, conv5_reshape is BxNxD, c[i, :] is D
    residual = conv5_reshape - c[i, :]
    # a is Bx1xN
    a = tf.expand_dims(conva_transpose[:, i, :], 1)
    # Bx1xN and BxNxD to Bx1xD
    if i == 0:
        V = tf.matmul(conva_transpose, residual)
    else:
        V = tf.concat([V, tf.matmul(conva_transpose, residual)], 1)

# intra-normalization
V = tf.nn.l2_normalize(V, dim=2)

# L2 normalization, output is a K x D discriptor
V = tf.nn.l2_normalize(V, dim=1)

# V: reshape V from BxKxD to Bx(KxD)
output = tf.reshape(V, [-1, K * D])
# print(output.get_shape())

with tf.name_scope("loss"):

    # get loss, dimension is batch_size
    for i in range(batch_size / 3):
        oriimg = output[i * 3]
        oriimg_aux1 = output[i * 3 + 1]
        oriimg_aux2 = output[i * 3 + 2]
        ed_ori2aux1 = tf.sqrt(tf.reduce_sum(tf.square(oriimg - oriimg_aux1)))
        ed_ori2aux2 = tf.sqrt(tf.reduce_sum(tf.square(oriimg - oriimg_aux2)))
        s = [ed_ori2aux1 - ed_ori2aux2 + 0.25]
        if i == 0:
            loss = tf.maximum(s, 0)
        else:
            loss = tf.concat([loss, tf.maximum(s, 0)], 0)

# Train op
with tf.name_scope("train"):

    # Create optimizer and apply gradient descent to the trainable variables
    train = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    t = sess.run(train, feed_dict={X: np.ones((batch_size, 227, 227, 3), dtype='float32')})
    print sess.run(loss, feed_dict={X: np.ones((batch_size, 227, 227, 3), dtype='float32')})