# coding=utf-8

import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow.contrib.slim as slim
import math
import os
import shutil
import gc


def get_initializer(weights, layer, w_or_b, stddev_num):
    pass


# padding = 'SAME' means WxH not change.
def conv(input, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1,
                                                            trainable=False, init_wb=None, initializer=None):
    """
    :param init_value: priority is greater
    :param initializer: priority is smaller
    :param padding: 'SAME' means output_shape is WxH, the same as input
    :return: net
    """

    # Get number of input channels
    input_channels = int(input.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        if init_wb:
            initializer = tf.constant_initializer(value=init_wb[name]['weights'],
                                                  dtype=tf.float32)
        weights = tf.get_variable(name='weights',
                                  shape=[filter_height, filter_width, input_channels / groups, num_filters],
                                  trainable=trainable,
                                  initializer=initializer)
        if init_wb:
            initializer = tf.constant_initializer(value=init_wb[name]['biases'],
                                      dtype=tf.float32)
        biases = tf.get_variable(name='biases',
                                 shape=[num_filters],
                                 trainable=trainable,
                                 initializer=initializer)

        if groups == 1:
            conv = convolve(input, weights)

        # In the cases of multiple groups, split inputs & weights and
        else:
            # Split input and weights and convolve them separately
            # input_groups = tf.split(3, groups, input)
            # weight_groups = tf.split(3, groups, weights)
            input_groups = tf.split(input, groups, 3)
            weight_groups = tf.split(weights, groups, 3)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            # conv = tf.concat(3, output_groups)
            conv = tf.concat(output_groups, 3)

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
def downsample(X):

    return X


# upsample Reweighting Mask m from Bx13x13x1 to BxWxHx1
def upsample(convw):

    return convw


# from every batch get top d refer imgs of all query imgs
def get_topd(query_vectors, refer_vectors, top, index_biase):

    for j in range(len(query_vectors)):
        # cal distance between query img and refer img
        d = np.array([np.sqrt(np.sum(np.square(query_vectors[j] - refer_vector))) for refer_vector in refer_vectors])
        temp_index = np.argsort(d)[0:top]
        temp_d = d[temp_index]
        temp_index += index_biase
        if j == 0:
            query_top_index = temp_index
            query_top_d = temp_d
        else:
            query_top_index = np.concatenate((query_top_index, temp_index))
            query_top_d = np.concatenate((query_top_d, temp_d))

    # dimension is len(query_url) x top
    query_top_index = query_top_index.reshape(-1, top)
    query_top_d = query_top_d.reshape(-1, top)
    return query_top_index, query_top_d


def get_topd_use_tf(query, refer, top, step, once_size):

    query_vectors = tf.placeholder(dtype=tf.float32, shape=[50, 512])
    # Bx1x(64*256)
    query_vectors_expdim = tf.expand_dims(query_vectors, 1)
    query_vectors_tile = tf.tile(query_vectors_expdim, [1, once_size, 1])
    refer_vectors = tf.placeholder(dtype=tf.float32, shape=[once_size, 512])
    # tf.get_variable(name='t', initializer=tf.constant(refer_vectors))
    refer_vectors_expdim = tf.expand_dims(refer_vectors, 0)
    refer_vectors_tile = tf.tile(refer_vectors_expdim, [50, 1, 1])
    # 3x108
    d = tf.sqrt(tf.reduce_sum(tf.multiply(query_vectors_tile - refer_vectors_tile, query_vectors_tile - refer_vectors_tile), axis=2))
    d = -d
    top_d, top_index = tf.nn.top_k(d, k=top, sorted=True)
    top_d = -top_d

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(query.shape[0] // 50):
            query_top = sess.run([top_d, top_index], feed_dict={query_vectors: query[i*50:(i+1)*50], refer_vectors: refer})
            if i == 0:
                query_top_d = query_top[0]
                query_top_index = query_top[1] + step * once_size
            else:
                query_top_d = np.concatenate((query_top_d, query_top[0]))
                query_top_index = np.concatenate((query_top_index, query_top[1] + step*once_size))
        left = query.shape[0] - 500 * query.shape[0] // 500
        query_top = sess.run([top_d, top_index],
                             feed_dict={query_vectors: query[left:], refer_vectors: refer})
        if i == 0:
            query_top_d = query_top[0]
            query_top_index = query_top[1] + step * once_size
        else:
            query_top_d = np.concatenate((query_top_d, query_top[0]))
            query_top_index = np.concatenate((query_top_index, query_top[1] + step * once_size))

    return query_top_d, query_top_index

    # with tf.Session() as sess:
    #     for i, query_vector in enumerate(query_vectors):
    #         query_vector = tf.tile(query_vector.reshape([1, -1]), [once_size, 1])
    #         d = tf.sqrt(tf.reduce_sum(tf.mul(query_vector - refer_vectors, query_vector - refer_vectors), axis=1))
    #         d = -d
    #         top_d, top_index = tf.nn.top_k(d, k=top, sorted=True)
    #         top_d = -top_d
    #         if i == 0:
    #             query_top_d = top_d
    #             query_top_index = top_index
    #         else:
    #             query_top_index = tf.concat(0, [query_top_index, top_index])
    #             query_top_d = tf.concat(0, [query_top_d, top_d])
    #
    #     query_top_index = sess.run(query_top_index)
    #     query_top_index += step * once_size
    #     query_top_d = sess.run(query_top_d)
    #
    # # dimension is len(query_url) x top
    # query_top_index = query_top_index.reshape(-1, top)
    # query_top_d = query_top_d.reshape(-1, top)
    # return query_top_index, query_top_d


# get top refer vectors from 2 batch
def compare_topd(last_indexs, last_ds, now_indexs, now_ds, top):

    # dimension is len(query_url) x top
    last_indexs = last_indexs.tolist()
    last_ds = last_ds.tolist()
    now_indexs = now_indexs.tolist()
    now_ds = now_ds.tolist()

    stackds = []
    stackindexs = []
    for i in range(len(last_indexs)):
        last_index = last_indexs[i]
        last_d = last_ds[i]
        now_index = now_indexs[i]
        now_d = now_ds[i]

        stackd = []
        stackindex = []
        while len(stackd) < top:
            flag = [0] * top
            for j in range(top):
                d = last_d[j]
                index = last_index[j]
                l = len(stackd)
                for k in range(top):
                   if now_d[k] < d and flag[k] == 0:
                       if len(stackd) == top:
                           break
                       stackd.append(now_d[k])
                       stackindex.append(now_index[k])
                       flag[k] = 1
                if len(stackd) == l:
                    if len(stackd) == top:
                        break
                    stackd.append(d)
                    stackindex.append(index)

        stackds.append(stackd)
        stackindexs.append(stackindex)

    return np.array(stackindexs), np.array(stackds)

    # # sort
    # return np.array(sorted(stackd)), np.array(sorted(stackindex))


# read img from url
def get_img(url):
    img = Image.open(url)
    img_resize = img.resize((227, 227), Image.ANTIALIAS)
    img2nda = np.array(img_resize)
    return img2nda


def get_imgs_by_size(dir, urls, step, once_size):

    container = []
    for j in range(once_size):
        url = dir + urls[step * once_size + j][:-1]
        img = get_img(url)
        container.append(img)
    refer_imgs = np.array(container)

    del container
    gc.collect()

    return refer_imgs


# get train and val steps
def get_step(filename, batch_size):
    with open(filename) as f:
        data = f.readlines()
    # get num of step
    temp = len(data) // (batch_size * 4)
    num_step = [temp + 1, temp][len(data) % (batch_size * 4) == 0]
    return num_step


# get one batch from data
def get_one_batch(filename, batch_size, step):

    # get traindata, sample '/streetview/PCI_sp_12017_37.785232_-122.392545_937762257_1_671121939_312.929_-0.510119.jpg'
    # get valdata, sample '/streetview/PCI_sp_12017_37.785232_-122.392545_937762257_1_671121939_312.929_-0.510119.jpg'

    with open(filename) as f:
        data = f.readlines()

    # data in train_file is grouped by 4 consist of 1 query img, 1 positive img, 2 negative imgs

    x_32 = []

    # start = abs(global_step * batch_size * 4 - (epoch - 1) * len(data))
    # for i in range(batch_size * 4):
    #     img_url = data[start + i if (start + i) < len(data) else (start + i - len(data))][0:-1]

    for i in range(batch_size * 4):
        img_url = data[step * batch_size * 4 + i][0:-1]
        url = 'Image-Geo-localization/train_data' + img_url
        # url = 'C:/Users/董鹏熙/Desktop/Image Geo-localization data/train_data' + img_url
        img = Image.open(url)
        img_resize = img.resize((227, 227), Image.ANTIALIAS)
        img2nda = np.array(img_resize)

        x_32.append(img2nda)

    # one group of 4 imgs will turn out 2 triplets, get batch_size triplets from x_32
    x_2 = []
    for i in range(batch_size):
        x_2.append(x_32[i * 4 + 0])
        x_2.append(x_32[i * 4 + 1])
        x_2.append(x_32[i * 4 + 2])
        x_2.append(x_32[i * 4 + 0])
        x_2.append(x_32[i * 4 + 1])
        x_2.append(x_32[i * 4 + 3])

    # get one batch of numpy array, shape is [batch_size * 3, 227, 227, 3].
    x_batch = np.array(x_2)
    return x_batch


def compute_crow_spatial_weight(X, a=2, b=2):
    """
    Given a tensor of features, compute spatial weights as normalized total activation.
    Normalization parameters default to values determined experimentally to be most effective.
    :param ndarray X:
        3d tensor of activations with dimensions (channels, height, width)
    :param int a:
        the p-norm
    :param int b:
        power normalization
    :returns ndarray:
        a spatial weight matrix of size (height, width)
    """
    S = X.sum(axis=0)
    z = (S**a).sum()**(1./a)
    return (S / z)**(1./b) if b != 1 else (S / z)


def compute_crow_channel_weight(X):
    """
    Given a tensor of features, compute channel weights as the
    log of inverse channel sparsity.
    :param ndarray X:
        3d tensor of activations with dimensions (channels, height, width)
    :returns ndarray:
        a channel weight vector
    """
    K, w, h = X.shape
    area = float(w * h)
    nonzeros = np.zeros(K, dtype=np.float32)
    for i, x in enumerate(X):
        nonzeros[i] = np.count_nonzero(x) / area

    nzsum = nonzeros.sum()
    for i, d in enumerate(nonzeros):
        nonzeros[i] = np.log(nzsum / d) if d > 0. else 0.

    return nonzeros


def apply_crow_aggregation(X):
    """
    Given a tensor of activations, compute the aggregate CroW feature, weighted
    spatially and channel-wise.
    :param ndarray X:
        3d tensor of activations with dimensions (channels, height, width)
    :returns ndarray:
        CroW aggregated global image feature
    """
    S = compute_crow_spatial_weight(X)
    C = compute_crow_channel_weight(X)
    X = X * S
    X = X.sum(axis=(1, 2))
    return X * C


def get_matched_imgs(refer_url, indexes, dir):
    """
    get matched imgs from 'last_query_top_index.txt'
    :param refer_url: streetview dataset
    :param indexes: top K matched img indexes
    :param dir: directory of streetview dataset
    :return:
    """
    # datatxt_dir = 'data/'
    # eval_refer_url_file = datatxt_dir + 'datatxt/extract_sanfran_sv_featext_fr.txt'
    # with open(eval_refer_url_file) as f:
    #     refer_url = np.array(f.readlines())

    # indexes = np.loadtxt('last_query_top_index.txt')

    if not os.path.exists('matched_imgs'):
        os.mkdir('matched_imgs')

    for i in range(indexes.shape[0]):
        if not os.path.exists(str(i)):
            os.mkdir(str(i))
        for j, index in enumerate(indexes[i]):
            # dir = 'C:/Users/董鹏熙/Desktop/Image Geo-localization data/train_data'
            img_url = dir + refer_url[int(index)][:-1]
            shutil.copy(img_url, 'matched_imgs/' + str(i))
            print('query img {}, index {}'.format(i, j))


def checkpoint2weights_npyfile():
    pass


def get_query_vectors():
    pass


def get_correct(query_urls, refer_urls, groundtruths, top, last_query_top_index):

    imgUrls = []
    for r in range(len(query_urls)):
        groundtruth = groundtruths[r // 3]
        for j in range(top):
            flag = 0
            for gt in groundtruth:
                if gt in refer_urls[last_query_top_index[r][j]]:
                    imgUrls.append(query_urls[r])
                    flag = 1
                    break
            if flag == 1:
                break

    return imgUrls