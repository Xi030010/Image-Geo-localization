#coding:utf-8

import tensorflow as tf
import numpy as np
from acn import ACN
import os
from datetime import datetime

# Path to the textfiles for the trainings and validation set
train_file = 'train.txt'
val_file = 'val.txt'

# Learning params
learning_rate = 0.001
momentum = 0.9
num_epochs = 10
batch_size = 24

# Network params
train_layers = ['convs', 'c', 'V']

# How often we want to write the tf.summary data to disk
display_step = 1

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = "C:/Users/董鹏熙/Desktop/Image-Geo-localization-master/finetune_alexnet/dogs_vs_cats"
checkpoint_path = "C:/Users/董鹏熙/Desktop/Image-Geo-localization-master/finetune_alexnet/"
# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

# input, we assume that input x is ordered by [Original, Original_aux1, Original_aux2]
x = tf.placeholder(tf.float32, [None, 227, 227, 3])
y = tf.placeholder(tf.string, [None])
# Initialize model
model = ACN(x, train_layers)
# link output to model output, dimension is Bx(KxD)
output = model.output

# # store the output images vector from all training batches
# refervectors = np.array([])
# refervectors = np.append(refervectors, tf.Session().run(vector))

# Op for calculating the loss
with tf.name_scope("loss"):

    for i in range(24):
        oriimg = output[i * 3]
        oriimg_aux1 = output[i * 3 + 1]
        oriimg_aux2 = output[i * 3 + 2]
        ed_ori2aux1 = tf.sqrt(tf.reduce_sum(tf.square(oriimg - oriimg_aux1)))
        ed_ori2aux2 = tf.sqrt(tf.reduce_sum(tf.square(oriimg - oriimg_aux2)))
        s = tf.Variable([ed_ori2aux1 - ed_ori2aux2 + 0.25])
        if i == 0:
            loss = tf.maximum(s, 0)
        else:
            loss = tf.concat(0, [loss, tf.maximum(s, 0)])

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients)

# Add gradients to summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

# Add the variables we train to the summary
for var in var_list:
    tf.summary.histogram(var.name, var)

# Add the loss to summary
tf.summary.scalar('cross_entropy', loss)

# Evaluation op: Accuracy of the model
# with tf.name_scope("accuracy"):
#     # get the best retrieved image from database
#     retrivector = np.array([])
#     for qvector in vector:
#         eds = np.array([])
#         for rvector in refervectors:
#             eds = np.append(eds, tf.sqrt(tf.reduce_sum(tf.square(qvector - rvector))))
#         minindex = np.argmax(eds)
#         retrivector = np.append(retrivector, refervectors(minindex))# 考虑数据类型；应该不只存query图片的vector，都要存
#
#     correct_img = tf.equal(retrivector, y)
#     accuracy = tf.reduce_mean(tf.cast(correct_img, tf.float32))

# Add the accuracy to the summary
# tf.summary.scalar('accuracy', accuracy)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

# Initialize the FileWriter
writer = tf.summary.FileWriter(filewriter_path)

# Initialize an saver for store model checkpoints
saver = tf.train.Saver()

# Get the batch data from dataset
pass

# Start tensorflow session
with tf.Session() as sess:

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # loop over epochs
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))

        step = 1

        # training
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            sess.run(train_op, feed_dict={x: np.array()})