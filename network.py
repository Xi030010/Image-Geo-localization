# coding=utf-8

import tensorflow as tf
import numpy as np
import os
import math
import gc
import utils
from utils import conv, lrn, max_pool, downsample, upsample
import time
import tensorflow.contrib.slim as slim
import shutil


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('mode', 'train', 'train or test')
tf.app.flags.DEFINE_string('trainlayers', 'conv3 conv4 conv5', 'train layers')
tf.app.flags.DEFINE_boolean('restore', False, 'restore model for train or test')
tf.app.flags.DEFINE_integer('restore_model_epoch', 0, 'epoch of restored model')
tf.app.flags.DEFINE_integer('restore_model_step', 0, 'step of restored model')
tf.app.flags.DEFINE_boolean('with_crn', False, 'if use crn')
tf.app.flags.DEFINE_boolean('with_frn', False, 'if use frn')
tf.app.flags.DEFINE_integer('K', 64, 'cluster num')
tf.app.flags.DEFINE_integer('output_dim', 8192, 'length of image representation vector')
tf.app.flags.DEFINE_integer('num_epoch', 30, 'num of training epoch')
tf.app.flags.DEFINE_integer('save_step', 1000, 'save model every num steps')
tf.app.flags.DEFINE_integer('batch_size', 24, 'batch size')
tf.app.flags.DEFINE_boolean('add_batch_norm', False, 'if add batch normalization')
tf.app.flags.DEFINE_string('CNN_model', 'alexnet', 'which CNN model to use')
tf.app.flags.DEFINE_string('weights_file', 'weights/alexnet_conv5_vlad_preL2_intra_white_pitts30k.npy', 'weights file')
tf.app.flags.DEFINE_integer('top', 5, 'top retrieved images')
tf.app.flags.DEFINE_string('datatxt_dir', '/home/ultron/dongpengxi/', 'direction of train and test data .txt file')
tf.app.flags.DEFINE_string('model_dir', 'NetVLAD_alexnet_finetuned_on_SF', 'direction of model to save and restore')


def alexnet(X, trainlayers=[], weights=None):

    # current = X
    # layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    # for layer in layers:
    #     if 'conv' in layer:
    #         pass

    # 1st Layer: Conv (w ReLu) -> Lrn -> Poolwith 1 groups

    trainable = False
    if 'conv1' in trainlayers:
        # and, or, weights==None and init_wb=weights['conv1'] or None
        # conv1 = conv(X, 11, 11, 96, 4, 4, padding='VALID', name='conv1', trainable=True, init_wb=weights)
        trainable = True
    conv1 = conv(X, 11, 11, 96, 4, 4, padding='VALID', name='conv1', init_wb=weights, trainable=trainable)
    norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
    pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID', name='pool1')

    # 2nd Layer: Conv (w ReLu) -> Lrn -> Poolwith 2 groups
    trainable = False
    if 'conv2' in trainlayers:
        trainable = True
    conv2 = conv(pool1, 5, 5, 256, 1, 1, groups=2, name='conv2', init_wb=weights, trainable=trainable)
    norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
    pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID', name='pool2')

    # 3rd Layer: Conv (w ReLu)
    trainable = False
    if 'conv3' in trainlayers:
        trainable = True
    conv3 = conv(pool2, 3, 3, 384, 1, 1, name='conv3',
                        trainable=trainable,
                        init_wb=weights)

    # 4th Layer: Conv (w ReLu) splitted into two groups
    trainable = False
    if 'conv4' in trainlayers:
        trainable = True
    conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4',
                        trainable=trainable,
                        init_wb=weights)

    # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
    trainable = False
    if 'conv5' in trainlayers:
        trainable = True
    conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5',
                        trainable=trainable,
                        init_wb=weights)

    return conv5


def vggnet(X, trainlayers=[], weights_file=None):

    layers = {'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
              'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
              'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
              'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
              'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3'}

    current = X
    if not weights_file:
        weights = np.load(weights_file).items()
    for i, layer in enumerate(layers):
        if 'conv' in layer:
            if layer in trainlayers:
                trainable = True
            kernels = utils.get_variable(weights[layer]['weights'], name=layer + '/weights', trainable=trainable)
            biases = utils.get_variable(weights[layer]['biases'], name=layer + '/biases', trainable=trainable)
            current = utils.conv2d(current, kernels, biases, name=layer)
        elif 'relu' in layer:
            current = tf.nn.relu(current, name=layer)
        elif 'pool' in layer:
            current = utils.avg_pool_2x2(current, name=layer)

    return current


def crn(conv5, weights_file):

    # Contextual Reweighting Network

    conv5 = downsample(conv5)
    # g Multiscale Context Filters, dimension is Bx13x13x84
    convg3x3 = conv(conv5, 3, 3, 32, 1, 1, name='convg3x3', trainable=True,
                    initializer=tf.contrib.layers.xavier_initializer())
    convg5x5 = conv(conv5, 5, 5, 32, 1, 1, name='convg5x5', trainable=True,
                    initializer=tf.contrib.layers.xavier_initializer())
    convg7x7 = conv(conv5, 7, 7, 20, 1, 1, name='convg7x7', trainable=True,
                    initializer=tf.contrib.layers.xavier_initializer())
    # convg = tf.concat([convg3x3, convg5x5, convg7x7], 3)
    convg = tf.concat(3, [convg3x3, convg5x5, convg7x7])
    # w Accumulation Weight, 13x13x84 to 13x13x1
    convw = conv(convg, 1, 1, 1, 1, 1, name='convw', trainable=True, initializer=tf.contrib.layers.xavier_initializer())
    # Bx13x13x1 to BxWxHx1
    m = upsample(convw)

    return m


def frn(featuremap, W, H, D):

    # CROW pooling

    D_sqrt = int(math.sqrt(D))
    # transpose feature from BxWxHxD to BxDxWxH
    CROW = tf.transpose(featuremap, [0, 3, 1, 2])
    # reshape feature from BxDxWxH to BxDsxDsx(WxH)
    CROW = tf.reshape(CROW, shape=[-1, D_sqrt, D_sqrt, W*H])
    # BxDsxDsx(WxH) to BxDsxDsx(WxH)
    convd3x3 = conv(CROW, 3, 3, W*H, 1, 1, name='convd3x3', padding='SAME', trainable=True)
    # BxDsxDsx(WxH) to BxDsxDsx1
    convd = conv(convd3x3, 1, 1, W*H, 1, 1, name='convd', padding='SAME', trainable=True)
    # BxDsxDsx1 to BxD
    convd = tf.reshape(convd, [-1, D])
    convd = tf.nn.softmax(convd)
    # BxD multiply BxWxHxD
    convd_tile = tf.tile(convd, [1, W, H, 1])
    featuremap_with_Dweights = tf.multiply(convd_tile, featuremap)

    return featuremap_with_Dweights


def siamesenet(D, loss_vlad):
    # # 2-channel Siamese Network
    # query_vector = tf.reshape(tf.Variable([vector[i * 3] for i in range(batch_size)]), [-1, K, D])
    # refer_p_vector = tf.reshape(tf.Variable([vector[i * 3 + 1] for i in range(batch_size)]), [-1, K, D])
    # refer_n_vector = tf.reshape(tf.Variable([vector[i * 3 + 2] for i in range(batch_size)]), [-1, K, D])
    # # merge to 2-channel, shape is [batch_size, K, 2*D]
    # qp = tf.concat(-1, [query_vector, refer_p_vector])
    # qn = tf.concat(-1, [query_vector, refer_n_vector])
    # Siamese_conv1 = conv(qp, 3, 3, 256, 1, 1, groups=2, name='Seamese_conv1')

    Siamese_x = tf.placeholder(tf.float32, [None, FLAGS.K, 2*D])
    Siamese_conv1 = conv(Siamese_x, 4, 4, 256, 1, 1, name='Siamese_conv1')
    Siamese_conv2 = conv(Siamese_conv1, 3, 3, 128, 1, 1, name='Siamese_conv2')
    Siamese_conv3 = conv(Siamese_conv2, 3, 3, 64, 1, 1, name='Siamese_conv3')
    Siamese_conv4 = conv(Siamese_conv3, 3, 3, 32, 1, 1, name='Siamese_conv4')
    Siamese_conv5 = conv(Siamese_conv4, 3, 3, 16, 1, 1, name='Siamese_conv5')
    Siamese_conv6 = conv(Siamese_conv5, 3, 3, 8, 1, 1, name='Siamese_conv6')
    Siamese_conv7 = conv(Siamese_conv6, 3, 3, 4, 1, 1, name='Siamese_conv7')
    Siamese_fc1 = utils.fc(Siamese_conv7, FLAGS.K*D*4, 512, name='Siamese_fc1')
    Siamese_fc2 = utils.fc(Siamese_fc1, 512, 1, name='Siamese_fc2')

    # loss_siamesenet =

    # return loss_siamesenet


def context_gating(input_layer):
    """Context Gating
    Args:
    input_layer: Input layer in the following shape:
    'batch_size' x 'number_of_activation'
    Returns:
    activation: gated layer in the following shape:
    'batch_size' x 'number_of_activation'
    """

    input_dim = input_layer.get_shape().as_list()[1]

    gating_weights = tf.get_variable("gating_weights",
                                     [input_dim, input_dim],
                                     initializer=tf.random_normal_initializer(
                                         stddev=1 / math.sqrt(input_dim)))

    gates = tf.matmul(input_layer, gating_weights)

    if FLAGS.add_batch_norm:
        gates = slim.batch_norm(
            gates,
            center=True,
            scale=True,
            is_training=FLAGS.is_training,
            scope="gating_bn")
    else:
        gating_biases = tf.get_variable("gating_biases",
                                        [input_dim],
                                        initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(input_dim)))
        gates += gating_biases

    gates = tf.sigmoid(gates)

    activation = tf.multiply(input_layer, gates)

    return activation


def netvlad(X, trainlayers=[], with_crn=False, with_crow=False, weights_file=None, training_procedure='crn', num_step_train=0):

    # read weights if weights is not None
    if weights_file:
        weights = np.load(weights_file, encoding='latin1').tolist()
    else:
        weights = None

    # 5 conv of alexnet
    with tf.name_scope('alexnet'):

        with tf.variable_scope('alexnet'):

            conv5 = alexnet(X=X,
                            trainlayers=trainlayers,
                            weights=weights)

    # contextual reweighting network
    with tf.name_scope('crn'):

        with tf.variable_scope('crn'):

            if with_crn:
                m = crn(conv5, weights_file)
                m = tf.expand_dims(m, -1)
                m = tf.tile(m, [1, 1, 1, FLAGS.K])

    # netvlad
    with tf.name_scope('netvlad'):

        with tf.variable_scope('netvlad'):

            # get W, H, D from featuremap
            # D is 256
            D = conv5.get_shape()[3].value
            # W is 13
            W = conv5.get_shape()[1].value
            # H is 13
            H = conv5.get_shape()[2].value

            # convs weights
            with tf.variable_scope('convs'):

                initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(D))
                if not weights:
                    if 'convs' in weights.keys():
                        initializer = tf.constant_initializer(value=weights['convs']['weights'], dtype=tf.float32)
                convs_weights = tf.get_variable('weights', [D, FLAGS.K], initializer=initializer)

                # reshaped_conv5 is (BxWxH) x D
                reshaped_conv5 = tf.reshape(conv5, shape=[-1, D])
                # (BxWxH) x D  matmul  DxK = (BxWxH) x K
                activation = tf.matmul(reshaped_conv5, convs_weights)

                # biase or batch normalization
                if FLAGS.add_batch_norm:
                    activation = slim.batch_norm(
                        activation,
                        center=True,
                        scale=True,
                        is_training=FLAGS.is_training,
                        scope="cluster_bn")
                else:
                    # convs biases
                    initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(D))
                    if not weights:
                        if 'convs' in weights.keys():
                            initializer = tf.constant_initializer(value=weights['convs']['biases'], dtype=tf.float32)
                    convs_biases = tf.get_variable('biases', [FLAGS.K], initializer=initializer)
                    activation = tf.nn.bias_add(activation, convs_biases)

            # get a by softmax, a is BxWxHxK
            activation = tf.nn.softmax(activation)

            # activation reshaped to Bx(WxH)xK
            activation = tf.reshape(activation, [-1, W*H, FLAGS.K])

            # use crn m
            if with_crn:
                activation = tf.multiply(activation, m)

            # a_sum is Bx1xK
            a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

            # clusters
            initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(D))
            if not weights:
                if 'clusters' in weights.keys():
                    initializer = tf.constant_initializer(value=weights['clusters'], dtype=tf.float32)
            clusters = tf.get_variable(name='clusters', shape=[1, D, FLAGS.K], initializer=initializer)

            # Bx1xK  multiply  1xDxK = BxDxK
            a = tf.multiply(a_sum, clusters)

            # BxKx(WxH)
            activation = tf.transpose(activation, perm=[0, 2, 1])

            # Bx(W*H)xD
            reshaped_conv5 = tf.reshape(reshaped_conv5, [-1, (W*H), D])

            # BxKx(WxH) matmul Bx(WxH)xD = BxKxD
            vlad = tf.matmul(activation, reshaped_conv5)
            # BxDxK
            vlad = tf.transpose(vlad, perm=[0, 2, 1])
            # BxDxK
            vlad = tf.subtract(vlad, a)

            vlad = tf.nn.l2_normalize(vlad, 1)

            # Bx(K*D)
            vlad = tf.reshape(vlad, [-1, FLAGS.K * D])
            vlad = tf.nn.l2_normalize(vlad, 1)

            # PCA weights and biases
            with tf.variable_scope('PCA'):

                if FLAGS.output_dim == FLAGS.K * D / 2:

                    initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(FLAGS.K))
                    if not weights:
                        if 'PCA' in weights.keys():
                            initializer = tf.constant_initializer(value=weights['PCA']['weights'], dtype=tf.float32)
                    PCA_weights = tf.get_variable(name='weights',
                                                  shape=[FLAGS.K*D, FLAGS.K * D / 2],
                                                  initializer=initializer,
                                                  trainable=True)

                    initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(FLAGS.K))
                    if not weights:
                        if 'PCA' in weights.keys():
                            initializer = tf.constant_initializer(value=weights['PCA']['biases'], dtype=tf.float32)
                    PCA_biases = tf.get_variable(name='biases', shape=[FLAGS.K * D / 2],
                                                 initializer=initializer, trainable=True)

                    # Bxoutput_dim
                    vlad = tf.matmul(vlad, PCA_weights)
                    vlad = tf.nn.bias_add(vlad, PCA_biases)
                else:
                    PCA_weights = tf.get_variable(name='weights',
                                                  shape=[FLAGS.K * D, FLAGS.output_dim],
                                                  initializer=initializer,
                                                  trainable=True)
                    PCA_biases = tf.get_variable(name='biases', shape=[FLAGS.output_dim], initializer=initializer,
                                                 trainable=True)
                    # Bxoutput_dim
                    vlad = tf.matmul(vlad, PCA_weights)
                    vlad = tf.nn.bias_add(vlad, PCA_biases)

                vector = vlad

                # context gating
                # vector = context_gating(vlad)

    # loss
    with tf.name_scope('loss'):

        # get loss, dimension is batch_size
        for i in range(FLAGS.batch_size * 2):
            oriimg = vector[i * 3]
            oriimg_p = vector[i * 3 + 1]
            oriimg_n = vector[i * 3 + 2]
            ed_ori2aux1 = tf.sqrt(tf.reduce_sum(tf.square(oriimg - oriimg_p)))
            ed_ori2aux2 = tf.sqrt(tf.reduce_sum(tf.square(oriimg - oriimg_n)))
            s = [ed_ori2aux1 - ed_ori2aux2 + 0.25]
            if i == 0:
                loss = tf.maximum(s, 0)
            else:
                loss = tf.concat([loss, tf.maximum(s, 0)], 0)
                # loss = tf.concat(0, [loss, tf.maximum(s, 0)])

    # Train op
    with tf.name_scope('train'):

        # Create optimizer and apply gradient descent to the trainable variables
        if training_procedure == 'crn':
            # var_list consists of 'crn network weights'
            var_list = [var for var in tf.trainable_variables() if 'crn' in var.name]
            if len(var_list) == 0:
                # train = tf.train.MomentumOptimizer(0.0005, 0.9).minimize(loss)
                train = tf.train.MomentumOptimizer(0.0005, 0.9).minimize(loss)
            else:
                train_crn_layer = tf.train.MomentumOptimizer(0.005, 0.9).\
                    minimize(loss,
                             var_list=[var for var in tf.trainable_variables() if 'crn' in var.name])
                train_other_layer = tf.train.MomentumOptimizer(0.0005, 0.9).\
                    minimize(loss,
                             var_list=[var for var in tf.trainable_variables() if 'crn' not in var.name])
                train = tf.group(train_crn_layer, train_other_layer)
        elif training_procedure == 'netvlad':
            global_step = tf.Variable(0, trainable=False)
            start_learning_rate = 0.001
            decay_epoch_of_learning_rate = 5
            decay_rate_of_learning_rate = 0.5
            learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
                                                       decay_epoch_of_learning_rate * num_step_train,
                                                       decay_rate_of_learning_rate,
                                                       staircase=True)
            train = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss,
                                                                                   global_step=global_step)

    # evaluate
    with tf.name_scope('evaluate'):
        pass

    return vector, loss, train


def main(argv=None):
    # dir of data txt files
    # datatxt_dir = 'data/'
    datatxt_dir = FLAGS.datatxt_dir
    train_url_file = 'datatxt/extract_all_sanfran_netvlad_trn_fr_rmgsv_rmflickr3.txt'

    # get num of train step per epoch
    num_step_train = utils.get_step(train_url_file, FLAGS.batch_size)

    # dir to save model
    model_dir = FLAGS.model_dir
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    # netvlad, crn, frn
    X = tf.placeholder(tf.float32, [None, 227, 227, 3])
    vector, loss, train = netvlad(X,
                                  trainlayers=FLAGS.trainlayers.split(),
                                  training_procedure='crn',
                                  with_crn=FLAGS.with_crn,
                                  with_crow=FLAGS.with_frn,
                                  weights_file=FLAGS.weights_file,
                                  num_step_train=num_step_train)


    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver()

        # train and validation
        if FLAGS.mode == 'train':

            # print all variables and trainable variables
            print([var for var in tf.trainable_variables() if 'crn' not in var.name])
            print('global variables: ' + str([x.name for x in tf.global_variables()]))
            print('trainable variables: ' + str([x.name + str(x.get_shape()) for x in tf.trainable_variables()]))

            # continue training from exist ending place
            if FLAGS.restore:

                model_name = model_dir + '/epoch_' + str(FLAGS.restore_model_epoch) + '_step_' + str(FLAGS.restore_model_step)
                saver.restore(sess, model_name)

                print('continue trainging:')

                for left_step_train in range(FLAGS.restore_model_step + 1, num_step_train + 1):

                    start_time = time.time()

                    # train
                    print('epoch: ' + str(FLAGS.restore_model_epoch) + ', step: ' + str(left_step_train) + ', batch is ready.')

                    x_batch = utils.get_one_batch(train_url_file,
                                                  FLAGS.batch_size,
                                                  left_step_train - 1)

                    sess.run(train, feed_dict={X: x_batch})

                    # print loss
                    print(sess.run(loss, feed_dict={X: x_batch}))

                    print('use time: ' + str(time.time() - start_time))

                    # save model if step is a multiple of save_step
                    if left_step_train % FLAGS.save_step == 0 or left_step_train == num_step_train:
                        model_name = model_dir + '/epoch_' + str(FLAGS.restore_model_epoch) + '_step_' + str(left_step_train)
                        saver.save(sess, model_name)
                        print('model saved.')

            # train loop for all epochs
            for epoch in range(FLAGS.restore_model_epoch + 1, FLAGS.num_epoch + 1):

                for step_train in range(1, num_step_train + 1):

                    start_time = time.time()

                    # train
                    print('epoch: ' + str(epoch) + ', step: ' + str(step_train) + ', batch is ready.')

                    x_batch = utils.get_one_batch(train_url_file,
                                                  FLAGS.batch_size,
                                                  step_train - 1)

                    sess.run(train, feed_dict={X: x_batch})

                    # print loss
                    print(sess.run(loss, feed_dict={X: x_batch}))

                    # print time of one step
                    print('use time: ' + str(time.time() - start_time))

                    # save model if step is a multiple of 500
                    if step_train % FLAGS.save_step == 0 or step_train == num_step_train:
                        model_name = model_dir + '/epoch_' + str(epoch) + '_step_' + str(step_train)
                        saver.save(sess, model_name)
                        print('model saved.')

            # save variable values to a .npy file
            params = {}
            reader = tf.train.NewCheckpointReader(model_dir + 'epoch_' + str(FLAGS.num_epoch) + '_step_' + str(num_step_train))
            reader_keys = reader.get_variable_to_shape_map().keys()
            for key in reader_keys:
                if 'Momentum' in key:
                    continue
                else:
                    layer = key.split('/')[0]
                    varname = key.split('/')[1]
                    if layer not in params.keys():
                        params[layer] = {}
                    params[layer][varname] = reader.get_tensor(key)
            filename = 'netvlad.npy'
            if FLAGS.with_crn:
                filename = 'crn_' + filename
            if FLAGS.with_frn:
                filename = 'frn_' + filename
            np.save(filename, params)

        # evaluation
        else:

            # catalog
            # eval_query_dir = 'C:/Users/董鹏熙/Desktop/Image Geo-localization data/'
            # eval_refer_dir = 'C:/Users/董鹏熙/Desktop/Image Geo-localization data/train_data'
            eval_query_dir = datatxt_dir + 'Image-Geo-localization/evaluation_data'
            eval_refer_dir = datatxt_dir + 'Image-Geo-localization/train_data'

            eval_query_url_file = datatxt_dir + 'datatxt/extract_sanfran_q3_featext_fr.txt'
            eval_query_groundtruth_file = datatxt_dir + 'datatxt/cartoid_groundTruth_2014_04.txt'
            eval_refer_url_file = datatxt_dir + 'datatxt/extract_sanfran_sv_featext_fr.txt'

            # read query img file
            # sample, 'evaluation_data/0000.jpg'
            with open(eval_query_url_file) as f:
                query_urls = np.array(f.readlines())
            # sample, '0000 671172717'
            with open(eval_query_groundtruth_file) as f:
                groundtruths = f.readlines()
            groundtruths = [t.split()[1:] for t in groundtruths]
            # sample, '/streetview/PCI_sp_11841_37.789727_-122.386935_937762214_0_727224067_246.264_-37.8198.jpg'
            with open(eval_refer_url_file) as f:
                refer_urls = np.array(f.readlines())

            # load model
            model_name = model_dir + '/epoch_' + str(FLAGS.restore_model_epoch)\
                         + '_step_' + str(FLAGS.restore_model_step)
            # model_name = 'epoch_' + str(current_epoch) + '_step_' + str(current_train_step)
            saver.restore(sess, model_name)

            once_size = 1000

            # vectors_dir = '/media/ultron/0D53F42E56755822/dongpengxi/'
            # if not os.path.exists(vectors_dir + 'vectors_txt'):
            #     os.mkdir(vectors_dir + 'vectors_txt')

            # get query vector from model
            # query_vector = utils.get_query_vector()
            num_loop_query = len(query_urls) // once_size
            for i in range(num_loop_query):

                query_imgs = utils.get_imgs_by_size(eval_query_dir,
                                                    query_urls,
                                                    i,
                                                    once_size)

                if i == 0:
                    query_vectors = sess.run(vector, feed_dict={X: query_imgs})
                else:
                    query_vectors = np.concatenate((query_vectors, sess.run(vector, feed_dict={X: query_imgs})))

                print('vector of query img group ' + str(i + 1) + ' of ' + str(num_loop_query) + ' done.')

                # release memory
                del query_imgs
                gc.collect()

            # last step
            if len(query_urls) > (num_loop_query * once_size):

                query_imgs = utils.get_imgs_by_size(eval_query_dir,
                                                    query_urls,
                                                    num_loop_query,
                                                    len(query_urls) - once_size * num_loop_query)

                if 'query_vectors' in locals().keys():
                    query_vectors = np.concatenate((query_vectors,
                                                    sess.run(vector, feed_dict={X: query_imgs})))
                else:
                    query_vectors = sess.run(vector, feed_dict={X: query_imgs})

                print('vector of query img last group done.')

                # release memory
                del query_imgs
                gc.collect()

            # np.savetxt(vectors_dir + 'vectors_txt/query_vector', query_vectors)

            # get refer vector from model
            # num_loop_refer = len(refer_urls) // once_size
            num_loop_refer = 124
            for i in range(num_loop_refer):

                start_time = time.time()

                refer_imgs = utils.get_imgs_by_size(eval_refer_dir,
                                                    refer_urls,
                                                    i,
                                                    once_size)

                refer_vectors = sess.run(vector, feed_dict={X: refer_imgs})

                print('vector of refer img group ' + str(i + 1) + ' of ' + str(num_loop_refer) +
                      'done.Get vector time is ' + str(time.time() - start_time) + 's.')

                # compare once_size refer vector and all query vector
                now_query_top_index, now_query_top_d = utils.get_topd(query_vectors,
                                                                      refer_vectors,
                                                                      FLAGS.top,
                                                                      i * once_size)
                # now_query_top_d, now_query_top_index = utils.get_topd_use_tf(query_vectors, refer_vectors, top, i, once_size)
                print('get top img time from one batch is: ' + str(time.time() - start_time) + 's.')

                if i == 0:
                    last_query_top_index = now_query_top_index
                    last_query_top_d = now_query_top_d
                else:
                    last_query_top_index, last_query_top_d = utils.compare_topd(last_query_top_index,
                                                                                last_query_top_d,
                                                                                now_query_top_index,
                                                                                now_query_top_d,
                                                                                FLAGS.top)
                # print('last_query_top_d: '+ str(last_query_top_d.shape))
                print('last_query_top_d:\n' + str(last_query_top_d))
                print('last_query_top_index:\n' + str(last_query_top_index))

                # cal accuracy
                correctImgs = utils.get_correct(query_urls, refer_urls, groundtruths, FLAGS.top, last_query_top_index)
                print(len(correctImgs))

                # np.savetxt(vectors_dir + 'vectors_txt/refer_vector_' + str(i + 1), refer_vectors)

                print('vector of refer img group ' + str(i + 1) + ' of ' + str(num_loop_refer) +
                      'done.Get top img time is ' + str(time.time() - start_time) + 's.')

                # release memory
                del refer_imgs, refer_vectors
                gc.collect()

            # save
            if not os.path.exists('evaluation_accuracy'):
                os.mkdir('evaluation_accuracy')
            np.savetxt('evaluation_accuracy/last_query_top_index.txt', last_query_top_index)
            np.savetxt('evaluation_accuracy/last_query_top_d.txt', last_query_top_d)
            # get correct query images
            if not os.path.exists('/home/ultron/dongpengxi/correctImages'):
                os.mkdir('/home/ultron/dongpengxi/correctImages')
            with open('/home/ultron/dongpengxi/correctImages/file.txt', 'w') as f:
                for img in correctImgs:
                    f.write(img)
            for img in correctImgs:
                shutil.copy(eval_query_dir + img[:-1], '/home/ultron/dongpengxi/correctImages')

            # # last step
            # if len(refer_urls) > (num_loop_refer * once_size):
            #
            #     refer_imgs = utils.get_imgs_by_size(eval_refer_dir,
            #                                         refer_urls,
            #                                         num_loop_refer,
            #                                         len(refer_urls) - num_loop_refer * once_size)
            #
            #     refer_vectors = sess.run(vector, feed_dict={X: refer_imgs})
            #
            #     now_query_top_index, now_query_top_d = utils.get_topd(query_vectors,
            #                                                           refer_vectors,
            #                                                           FLAGS.top,
            #                                                           num_loop_refer * once_size)
            #     # now_query_top_d, now_query_top_index = utils.get_topd_use_tf(query_vectors, refer_vectors, top, i,
            #     #                                                               once_size)
            #
            #     last_query_top_index, last_query_top_d = utils.compare_topd(last_query_top_index,
            #                                                                 last_query_top_d,
            #                                                                 now_query_top_index,
            #                                                                 now_query_top_d,
            #                                                                 FLAGS.top)
            #
            #     np.savetxt(vectors_dir + 'vectors_txt/refer_vector_' + str(num_loop_refer + 1), refer_vectors)
            #
            #     print('vector of refer img ' + str(num_loop_refer + 1) + 'group done.')
            #
            #     # release memory
            #     del refer_imgs, refer_vectors
            #     gc.collect()

            # save
            # np.savetxt('evaluation_accuracy/last_query_top_index.txt', last_query_top_index)
            # np.savetxt('evaluation_accuracy/last_query_top_d.txt', last_query_top_d)

            # cal accuracy
            # print(utils.get_correct(query_urls, refer_urls, groundtruths, FLAGS.top, last_query_top_index))


if __name__ == '__main__':
    tf.app.run()