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
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(axis=3, values=output_groups)

        # Add biases
        bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())

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


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

class ACN(object):

    def __init__(self, x, skip_layer, keep_prob = 0, weights_path = 'DEFAULT'):

        """
        Inputs:
        :param x:
        :param keep_prob:
        :param skip_layer:
        :param weights_path:
        """

        # Parse input arguments
        self.X = x
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):

        # alexnet conv1 - conv5 that after Relu activation

        # 1st Layer: Conv (w ReLu) -> Lrn -> Pool
        conv1 = conv(self.X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
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
        # x -> s, WxHxD is conv5 output of AlexNet
        k_h = 1;
        k_w = 1;
        c_o = 64;
        s_h = 0;
        s_w = 0;
        convs = conv(conv5, k_h, k_w, c_o, s_h, s_w, padding="VALID", name="convs")
        # s -> a
        WxHxK = tf.nn.softmax(convs)
        # parameter ck, totally we have k cks.
        c = tf.Variable(tf.random_normal([64, 256]))  # 2-D python array

        # VLAD core, get vector V whose dimension is K x D.
        # Let's try to use a loop to assign V firstly.
        V = tf.Variable(tf.zeros([64, 256]))
        for k in range(64):
            for j in range(256):
                cc = tf.constant(-c[k][j])
                for w in range(13):
                    for h in range(13):
                        V[k][j] = tf.assign(V[k][j],
                                            tf.add(tf.add(V[k][j], tf.multiply(WxHxK[w][h][k], conv5[w][h][j])), cc))
        V_afterreshape = tf.reshape(V, [-1, 64*256])

        # intra-normalization
        V_afterintra = tf.nn.l2_normalize(V_afterreshape, dim=2)

        # L2 normalization, output is a K x D discriptor
        self.output = tf.nn.l2_normalize(V_afterintra, dim=1)

    def load_initial_weights(self, session):

        # Load the weights into memory
        weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if the layer is one of the layers that should be reinitialized
            if op_name not in self.SKIP_LAYER:

                with tf.variable_scope(op_name, reuse=True):

                    # Loop over list of weights/biases and assign them to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:

                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:

                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))