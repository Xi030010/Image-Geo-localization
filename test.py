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
            input_groups = tf.split(3, groups, x)
            weight_groups = tf.split(3, groups, weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(3, output_groups)

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

X = tf.placeholder(tf.float32, [None, 227, 227, 3])

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

# NetVLAD pooling layer, based on AlexNet

# soft_assignment
# x -> s, WxHxD is the dimension of conv5 output of AlexNet, and dimension of convs is WxHxK
k_h = 1
k_w = 1
c_o = 64
s_h = 1
s_w = 1
convs = conv(conv5, k_h, k_w, c_o, s_h, s_w, padding="VALID", name="convs")
# print(convs.get_shape())
# s -> a
conva = tf.nn.softmax(convs)
# parameter ck, totally we have k cks.The dimension of ck is KxD.
c = tf.Variable(tf.random_normal([256, 64]))  # 2-D python array

# VLAD core, get vector V whose dimension is K x D. Let's try to use a loop to assign V firstly.
# a: reshape a from BxWxHxK to BxNxK
conva_reshape = tf.reshape(conva, shape=[-1, 13 * 13, 64])
# a: transpose a from NxK to KxNxB
conva_transpose = tf.transpose(conva_reshape)
# c: expand c from DxK to WxHxDxK
c_expand = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(c, 0), [13, 1, 1]), 0), [13, 1, 1, 1])
# c_batch = tf.tile(tf.expand_dims(c_expand, 0), [])
# c:reshape c from WxHxDxK to NxDxK
c_reshape = tf.reshape(c_expand, [169, 256, 64])
# conv5: expand conv5 from BxWxHxD to BxWxHxDxK
conv5_expand = tf.tile(tf.expand_dims(conv5, -1), [1, 1, 1, 1, 64])
# conv5_reshape = tf.reshape(conv5, [13*13, 256])  #reshape conv5 from WxHxD to NxD
# conv5: reshape conv5 from BxWxHxDxK to BxNxDxK
conv5_reshape = tf.reshape(conv5_expand, [-1, 169, 256, 64])
# residuals: dimension of residuals is BxNxDxK
residuals = tf.subtract(conv5_reshape, c_reshape)
# get V whose dimension is BxKxD
for j in range(24):
    V = tf.Variable([])
    for i in range(64):
        if i == 0:
            # V is calculated by 1xN multiply NxD, and dimension of V is 1xD
            V = tf.matmul(
                    tf.reshape(
                        conva_transpose[i, :, j], [1, -1]), tf.reshape(residuals[j, :, :, i], [13*13, 256]))
        else:
            V = tf.concat(0, [V, tf.matmul(
                    tf.reshape(
                        conva_transpose[i, :, j], [1, -1]), tf.reshape(residuals[j, :, :, i], [13*13, 256]))])
    if j == 0:
        Va = V
    else:
        Va = tf.concat(0, [Va, V])

# KxVxK = tf.matmul(conva_transpose, tf.subtract(conv5_reshape, c_expand))  # KxDxK
# V = tf.Variable([], dtype='float32')
# for i in range(64):
#     V = tf.concat(0, [V, KxVxK[i, i]])  # KxD
# V = tf.Variable(tf.zeros([64, 256]))
# for k in range(64):
#     for j in range(256):
#         cc = tf.constant(-c[k][j])
#         for w in range(13):
#             for h in range(13):
#                 V[k][j] = tf.assign(V[k][j],
#                                     tf.add(tf.add(V[k][j], tf.multiply(conva[w][h][k], conv5[w][h][j])), cc))

# V: reshape V from KxD to 1x(KxD)
Va = tf.reshape(V, [1, -1])

# intra-normalization
Va = tf.nn.l2_normalize(V, dim=2)

# L2 normalization, output is a K x D discriptor
output = tf.nn.l2_normalize(V, dim=1)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    t = sess.run(output, feed_dict={X: np.ones((24, 227, 227, 3), dtype='float32')})
    print(t)