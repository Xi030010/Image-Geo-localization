import tensorflow as tf
import numpy as np
import os
from PIL import Image
import math
import gc
import utils

# padding = 'SAME' means WxH not change.
def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
                                initializer=None, padding='SAME', trainable=False, groups=1):

    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # if name == 'convs':
        #     weights = tf.get_variable('c')
        # else:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights',
                                  shape=[filter_height, filter_width,
                                         input_channels / groups, num_filters],
                                  trainable=trainable, initializer=initializer)
        biases = tf.get_variable('biases', shape=[num_filters], trainable=trainable)

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


# downsample conv5 from BxWxHxD to Bx13x13xD
def downsample(conv5):

    return conv5


# upsample Reweighting Mask m from Bx13x13x1 to BxWxHx1
def upsample(convw):

    return convw


# load weights from AlexNet
def load_initial_weights(sess, weights_path, train_layers=[]):

    # Load the weights into memory
    weights_dict = np.load(weights_path, encoding='bytes').item()

    # Loop over all layer names stored in the weights dict
    for op_name in weights_dict:

        # Check if the layer is one of the layers that should be reinitialized
        if op_name not in train_layers:

            if op_name in ['fc6', 'fc7', 'fc8']:
                continue

            with tf.variable_scope(op_name, reuse=True):
                # Loop over list of weights/biases and assign them to their corresponding tf variable
                for data in weights_dict[op_name]:

                    # Biases
                    if len(data.shape) == 1:
                        sess.run(tf.get_variable('biases').assign(data))
                    # Weights
                    else:
                        sess.run(tf.get_variable('weights').assign(data))


X = tf.placeholder(tf.float32, [None, 227, 227, 3])

K = 64

num_epoch = 10

batch_size = 18


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
convg3x3 = conv(conv5, 3, 3, 32, 1, 1, name='convg3x3', trainable=True, initializer=tf.contrib.layers.xavier_initializer())
convg5x5 = conv(conv5, 5, 5, 32, 1, 1, name='convg5x5', trainable=True, initializer=tf.contrib.layers.xavier_initializer())
convg7x7 = conv(conv5, 7, 7, 20, 1, 1, name='convg7x7', trainable=True, initializer=tf.contrib.layers.xavier_initializer())
convg = tf.concat(3, [convg3x3, convg5x5, convg7x7])
# w Accumulation Weight, 13x13x84 to 13x13x1
convw = conv(convg, 1, 1, 1, 1, 1, name='convw', trainable=True, initializer=tf.contrib.layers.xavier_initializer())
# Bx13x13x1 to BxWxHx1
m = upsample(convw)


# # CROW pooling
#
# D_sqrt = int(math.sqrt(D))
# # transpose conv5 from BxWxHxD to BxDxWxH
# CROW = tf.transpose(conv5, [0, 3, 1, 2])
# # reshape conv5 from BxDxWxH to BxDsxDsx(WxH)
# CROW = tf.reshape(CROW, shape=[-1, D_sqrt, D_sqrt, W*H])
# # BxDsxDsx(WxH) to BxDsxDsx(WxH)
# convd3x3 = conv(CROW, 3, 3, W*H, 1, 1, name='convd3x3', padding='SAME', trainable=True)
# # BxDsxDsx(WxH) to BxDsxDsx1
# convd = conv(convd3x3, 1, 1, W*H, 1, 1, name='convd', padding='SAME', trainable=True)
# # BxDsxDsx1 to BxD
# convd = tf.reshape(convd, [-1, D])
# # BxD multiply BxWxHxD
# convd_tile = tf.tile(convd, [1, W, H, 1])
# conv5_with_Dweights = tf.multiply(convd_tile, conv5)


# NetVLAD pooling layer, based on AlexNet

# parameter ck, totally we have k cks.The dimension of ck is KxD.
with tf.variable_scope('convs'):
    c = tf.get_variable('c', [K, D], trainable=True,
                        initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(W*H*D)))
# c = tf.Variable(tf.ones([K, D]))

# soft_assignment
# x -> s, BxWxHxD is the dimension of conv5 output of AlexNet, and dimension of convs is BxWxHxK
k_h = 1
k_w = 1
c_o = K
s_h = 1
s_w = 1
convs = conv(conv5, k_h, k_w, c_o, s_h, s_w, name="convs", trainable=True,
             initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(W*H*D)))
# convs = conv(conv5_with_Dweights, k_h, k_w, c_o, s_h, s_w, name="convs", trainable=True,
#              initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(W*H*D)))
# s -> a, a is BxWxHxK
conva = tf.nn.softmax(convs)

# tile m from BxWxHx1 to BxWxHxK
m_tile = tf.tile(m, [1, 1, 1, K])
# get conva multiply m
conva_with_m = tf.multiply(conva, m_tile)

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
    # Bx1xN matmul BxNxD to Bx1xD
    if i == 0:
        V = tf.matmul(a, residual)
    else:
        V = tf.concat(1, [V, tf.matmul(a, residual)])

# intra-normalization
V = tf.nn.l2_normalize(V, dim=2)

# L2 normalization, output is a B x K x D discriptor
V = tf.nn.l2_normalize(V, dim=1)

# V: reshape V from BxKxD to Bx(KxD)
vector = tf.reshape(V, [-1, K * D])


# 2-channel Siamese Network
# query_vector = tf.reshape(tf.Variable([vector[i * 3] for i in range(batch_size)]), [-1, K, D])
# refer_p_vector = tf.reshape(tf.Variable([vector[i * 3 + 1] for i in range(batch_size)]), [-1, K, D])
# refer_n_vector = tf.reshape(tf.Variable([vector[i * 3 + 2] for i in range(batch_size)]), [-1, K, D])
# # merge to 2-channel, shape is [batch_size, K, 2*D]
# qp = tf.concat(-1, [query_vector, refer_p_vector])
# qn = tf.concat(-1, [query_vector, refer_n_vector])
# Siamese_conv1 = conv(qp, 3, 3, 256, 1, 1, groups=2, name='Seamese_conv1')

# Siamese_x = tf.placeholder(tf.float32, [None, K, 2*D])
# Siamese_conv1 = conv(Siamese_x, 4, 4, 256, 1, 1, name='Siamese_conv1')
# Siamese_conv2 = conv(Siamese_conv1, 3, 3, 128, 1, 1, name='Siamese_conv2')
# Siamese_conv3 = conv(Siamese_conv2, 3, 3, 64, 1, 1, name='Siamese_conv3')
# Siamese_conv4 = conv(Siamese_conv3, 3, 3, 32, 1, 1, name='Siamese_conv4')
# Siamese_conv5 = conv(Siamese_conv4, 3, 3, 16, 1, 1, name='Siamese_conv5')
# Siamese_conv6 = conv(Siamese_conv5, 3, 3, 8, 1, 1, name='Siamese_conv6')
# Siamese_conv7 = conv(Siamese_conv6, 3, 3, 4, 1, 1, name='Siamese_conv7')
# Siamese_fc1 = fc(Siamese_conv7, K*D*4, 512, name='Siamese_fc1')
# Siamese_fc2 = fc(Siamese_fc1, 512, 1, name='Siamese_fc2')


with tf.name_scope("loss"):
    # get loss, dimension is batch_size
    for i in range(batch_size):
        oriimg = vector[i * 3]
        oriimg_p = vector[i * 3 + 1]
        oriimg_n = vector[i * 3 + 2]
        ed_ori2aux1 = tf.sqrt(tf.reduce_sum(tf.square(oriimg - oriimg_p)))
        ed_ori2aux2 = tf.sqrt(tf.reduce_sum(tf.square(oriimg - oriimg_n)))
        s = [ed_ori2aux1 - ed_ori2aux2 + 0.25]
        if i == 0:
            loss = tf.maximum(s, 0)
        else:
            loss = tf.concat(0, [loss, tf.maximum(s, 0)])

# Train op
with tf.name_scope("train"):
    # Create optimizer and apply gradient descent to the trainable variables
    train = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)


# get traindata, sample '/streetview/PCI_sp_12017_37.785232_-122.392545_937762257_1_671121939_312.929_-0.510119.jpg'
train_file = 'data/datatxt/extract_all_sanfran_netvlad_trn_fr_rmgsv_rmflickr3.txt'
with open(train_file) as f:
    traindata = f.readlines()
# get num of train step every epoch
temp = len(traindata) // (batch_size * 2)
num_step_train = [temp + 1, temp][len(traindata) % (batch_size * 2) == 0]

# get valdata, sample '/streetview/PCI_sp_12017_37.785232_-122.392545_937762257_1_671121939_312.929_-0.510119.jpg'
val_file = 'data/datatxt/extract_val_sanfran_netvlad_trn_fr_rm.txt'
with open(val_file) as f:
    valdata = f.readlines()
# get num of val step
temp = len(valdata) // (batch_size * 2)
num_step_val = [temp + 1, temp][len(valdata) % (batch_size * 2) == 0]
# get one batch from data
def get_one_batch(data, step):
    # data in train_file is grouped by 4 consist of 1 query img, 1 positive img, 2 negative imgs

    x_32 = []
    # if num of one batch < batch size, get rest of all
    # if len(data) - step * (batch_size * 2) >= batch_size * 2:
    #     batch_img_num = int(batch_size / 2 * 4)
    # else:
    #     batch_img_num = len(data) - step * (batch_size * 2)
    for i in range(batch_size * 2):
        img_url = data[step * batch_size * 2 + i][0:-1]
        if 'streetview' in img_url:
            url = 'C:/Users/董鹏熙/Desktop/Image Geo-localization data/train_data' + img_url
        else:
            url = 'C:/Users/董鹏熙/Desktop/Image Geo-localization data/train_data' + img_url
        img = Image.open(url)
        img_resize = img.resize((227, 227), Image.ANTIALIAS)
        img2nda = np.array(img_resize)

        x_32.append(img2nda)

    # one group of 4 imgs will turn out 2 triplets, get batch_size triplets from x_32
    x_2 = []
    for i in range(int(batch_size / 2)):
        x_2.append(x_32[i * 4 + 0])
        x_2.append(x_32[i * 4 + 1])
        x_2.append(x_32[i * 4 + 2])
        x_2.append(x_32[i * 4 + 0])
        x_2.append(x_32[i * 4 + 1])
        x_2.append(x_32[i * 4 + 3])

    # get one batch of numpy array, shape is [batch_size * 3, 227, 227, 3].
    x_batch = np.array(x_2)
    return x_batch


# # get val data, sample '/flickr3/09698151_13989403180.jpg 671671827 671672078'
# # get x_val data, sameple '/flickr3/09698151_13989403180.jpg'
# val_file = 'data/datatxt/arrangement_exist.txt'
# with open(val_file) as f:
#     valdata = f.readlines()
# valdata = [x.split() for x in valdata]
# def get_x_val():
#     x_val = []
#     for i in range(len(valdata)):
#         img = Image.open('data/train_data' + valdata[i][0])
#         img_resize = img.resize((227, 227), Image.ANTIALIAS)
#         img2nda = np.array(img_resize)
#
#         x_val.append(img2nda)
#     x_val = np.array(x_val)
#     return x_val
# # get y_val data, sample '671671827 671672078'
# def get_y_val():
#     return [x[1:] for x in valdata]


# is_train = True
is_train = False
with tf.Session() as sess:

    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver()

    model_dir = 'model'
    if not os.path.exists('model'):
        os.mkdir('model')

    # train and validation
    if is_train:

        # load_initial_weights(sess, 'bvlc_alexnet.npy')
        saver.restore(sess, 'model/epoch_0_step_8600')
        print('global variables: ' + str([x.name for x in tf.global_variables()]))
        print('trainable variables: ' + str([x.name + str(x.get_shape()) for x in tf.trainable_variables()]))

        # train loop for all epochs
        for epoch in range(num_epoch):

            for step_train in range(num_step_train):

                # train
                print('epoch: ' + str(epoch) + ', step: ' + str(step_train) + ', batch is ready.')
                x_batch = get_one_batch(traindata, step_train)

                sess.run(train, feed_dict={X: x_batch})

                # print loss
                print(sess.run(loss, feed_dict={X: x_batch}))
                # test = sess.run(vector, feed_dict={X: x_batch})
                # print(len([x for x in test[0] if x != 0]))

                # # save vectors, refervectors will grow like [onebatch, K*D] to [twobatch, K*D]
                # if step == 0:
                #     refervectors = sess.run(vector, feed_dict={X: x_batch})
                # else:
                #     # concatenate will concat 2 ndarrays on 1th axis
                #     refervectors = np.concatenate((refervectors, sess.run(vector, feed_dict={X: x_batch})))

                # save model if step is a multiple of 500
                if step_train % 100 == 0 or step_train == num_step_train - 1:
                    saver.save(sess, os.path.join(model_dir, 'epoch_' + str(epoch) + '_step_' + str(step_train)))
                    print('model saved.')

                # # validation
                # if step == 0 or step % 500 != 0:
                #     continue
                # with open('data/datatxt/extract_sanfran_sv_featext_fr.txt') as f:
                #     refer_url = np.array(f.readlines())
                # container = []
                # i = 0
                # for refer in refer_url:
                #     img = Image.open('C:/Users/董鹏熙/Desktop' + refer[:-1])
                #     img_resize = img.resize((227, 227), Image.ANTIALIAS)
                #     img2nda = np.array(img_resize)
                #     container.append(img2nda)
                #     i += 1
                #     if i % 1000 == 0:
                #         print('load img ' + str(i))
                # refer_img = np.array(container)
                # refer_vector = sess.run(vector, feed_dict={X: refer_img})
                #
                # x_val = get_x_val()
                # y_val = get_y_val()
                # val_vector = sess.run(vector, feed_dict={X: x_val})  # val_vector is a ndarray
                # correct = 0
                # for i in range(len(val_vector)):
                #
                #     query = val_vector[i]  # shape of query is K*D, type is ndarray
                #
                #     # find the min from refervectors, shape of d is len(refervectors)
                #     d = np.array([np.sqrt(np.sum(np.square(query - refer))) for refer in refer_vector])
                #     # for refer in refervectors:
                #     #     d.append(np.sqrt(np.sum(np.square(query - refer))))
                #     min_index = d.argmin()
                #     min_imgurl = refer_url[min_index]
                #
                #     # judge
                #     for y in y_val[i]:
                #         if y in min_imgurl:
                #             correct = correct + 1
                #             break
                #
                # # calculate accuracy
                # accuracy = correct / len(x_val)
                # print(accuracy)

                # validation
                if step_train == 0:
                    continue
                if step_train % 10000 == 0:

                    correct = 0
                    for step_val in range(num_step_val):

                        print(str(step_val) + 'in' + str(num_step_val))

                        val_batch = get_one_batch(valdata, step_val)

                        val_batch_loss = sess.run(loss, feed_dict={X: val_batch})

                        # if loss is 0, correct plus 1
                        for x in val_batch_loss:
                            if x == 0:
                                correct += 1

                        print(correct / ((step_val + 1) * batch_size))

                    print(correct / (num_step_val * batch_size))
                    with open('val_accuracy/valcorrect', 'a') as f:
                        f.write('epoch: ' + str(epoch) + ', step: ' + str(step_train) +
                                ', accuracy: ' + str(correct / (num_step_val * batch_size)) + '\n')

    # evaluation
    else:

        # read query img file
        # sample, 'evaluation_data/0000.jpg'
        with open('data/datatxt/extract_sanfran_q3_featext_fr.txt') as f:
            query_url = np.array(f.readlines())
        # sample, '0000 671172717'
        with open('data/datatxt/cartoid_groundTruth_2014_04.txt') as f:
            query_groundtruth = f.readlines()
        query_groundtruth = [t.split()[1:] for t in query_groundtruth]
        # sample, '/streetview/PCI_sp_11841_37.789727_-122.386935_937762214_0_727224067_246.264_-37.8198.jpg'
        with open('data/datatxt/extract_sanfran_sv_featext_fr.txt') as f:
            refer_url = np.array(f.readlines())


        # load model
        saver.restore(sess, 'model/epoch_0_step_3600')


        once_size = 108
        top = 5

        # get query vector from model
        if not os.path.exists('E:/eval_vector/query_vector'):
            num_loop_query = len(query_url) // once_size
            for i in range(num_loop_query):
                container = []
                for j in range(once_size):
                    img = Image.open('C:/Users/董鹏熙/Desktop/Image Geo-localization data/' + query_url[i * once_size + j][:-1])
                    img_resize = img.resize((227, 227), Image.ANTIALIAS)
                    img2nda = np.array(img_resize)
                    container.append(img2nda)
                query_img = np.array(container)
                if i == 0:
                    query_vector = sess.run(vector, feed_dict={X: query_img})
                else:
                    query_vector = np.concatenate((query_vector, sess.run(vector, feed_dict={X: query_img})))
                print('vector of query img group ' + str(i + 1) + ' of ' + str(num_loop_query) + 'done.')

            if len(query_url) > (num_loop_query * once_size):
                container = []
                for i in range(len(query_url) - num_loop_query * once_size):
                    img = Image.open(
                        'C:/Users/董鹏熙/Desktop/Image Geo-localization data/' + query_url[num_loop_query * once_size + i][:-1])
                    img_resize = img.resize((227, 227), Image.ANTIALIAS)
                    img2nda = np.array(img_resize)
                    container.append(img2nda)
                query_img = np.array(container)
                query_vector = np.concatenate((query_vector,
                                               sess.run(vector, feed_dict={X: query_img})))
                print('vector of query img last group done.')

            # np.savetxt('E:/eval_vector/query_vector', query_vector)

            # release memory
            del container, query_img
            gc.collect()
        else:
            print('vector of query img exist.')

        # get refer vector from model
        num_loop_refer = len(refer_url) // once_size
        last_query_top_index = np.array([])
        last_query_top_d = np.array([])
        for i in range(num_loop_refer):
            if not os.path.exists('E:/eval_vector/refer_vector_' + str(i + 1)):
                container = []
                for j in range(once_size):
                    img = Image.open('C:/Users/董鹏熙/Desktop/Image Geo-localization data/train_data' + refer_url[i * once_size + j][:-1])
                    img_resize = img.resize((227, 227), Image.ANTIALIAS)
                    img2nda = np.array(img_resize)
                    container.append(img2nda)
                refer_img = np.array(container)
                # if i == 0:
                #     refer_vector = sess.run(vector, feed_dict={X: refer_img})
                # else:
                #     refer_vector = np.concatenate((refer_vector, sess.run(vector, feed_dict={X: refer_img})))
                refer_vector = sess.run(vector, feed_dict={X: refer_img})

                # compare once_size refer vector and all query vector
                now_query_top_index, now_query_top_d = utils.get_topd(query_vector, refer_vector, top)

                last_query_top_index, last_query_top_d = utils.compare_topd(last_query_top_index, last_query_top_d,
                                                                          now_query_top_index, now_query_top_d, top)

                # np.savetxt('E:/eval_vector/refer_vector_' + str(i + 1), refer_vector)
                print('vector of refer img group ' + str(i + 1) + ' of ' + str(num_loop_refer) + 'done.')

                # release memory
                del container, refer_img, refer_vector
                gc.collect()
            else:
                print('vector of refer img group ' + str(i + 1) + ' of ' + str(num_loop_refer) + 'exist.')

        if len(refer_url) > (num_loop_refer * once_size):
            if not os.path.exists('E:/eval_vector/refer_vector_' + str(num_loop_refer + 1)):
                for i in range(len(refer_url) - num_loop_refer * once_size):
                    img = Image.open('C:/Users/董鹏熙/Desktop/Image Geo-localization data/train_data' + refer_url[num_loop_refer * once_size + i][:-1])
                    img_resize = img.resize((227, 227), Image.ANTIALIAS)
                    img2nda = np.array(img_resize)
                    container.append(img2nda)
                refer_img = np.array(container)
                refer_vector = sess.run(vector, feed_dict={X: refer_img})
                np.savetxt('E:/eval_vector/refer_vector_' + str(num_loop_refer + 1), refer_vector)
                print('vector of refer img ' + str(num_loop_refer + 1) + 'group done.')

                # release memory
                del container, refer_img, refer_vector
                gc.collect()
            else:
                print('vector of refer img last group exist.')

        # ground truth match query_match
        correct = 0
        for i in range(len(query)):
            url = refer_url[query_match[i]]
            for groundtruth in query_groundtruth[i // 3]:
                # match
                if groundtruth in url:
                    correct += 1
                    break
            print(str(correct / (i + 1)))
        accuracy = correct / len(query)
        print(accuracy)