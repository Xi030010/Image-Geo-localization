#coding:utf-8

import tensorflow as tf
import numpy as np
from acn import ACN
import os
from datetime import datetime
from PIL import Image

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

# Path for tf.summary.FileWriter and to store model checkpoints in different os
if os.name == 'nt':
    filewriter_path = "C:/Users/董鹏熙/Desktop/Image-Geo-localization-master/finetune_alexnet/dogs_vs_cats"
    checkpoint_path = "C:/Users/董鹏熙/Desktop/Image-Geo-localization-master/finetune_alexnet/"
elif os.name == 'posix':
    filewriter_path = "/Users/dongpengxi/Desktop/Image Geo-localization/finetune_alexnet/filewriter"
    checkpoint_path = "/Users/dongpengxi/Desktop/Image Geo-localization/finetune_alexnet"
# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path): os.mkdir(checkpoint_path)

# input, we assume that input x is ordered by [Original, Original_aux1, Original_aux2]
x = tf.placeholder(tf.float32, [None, 227, 227, 3])
y = tf.placeholder(tf.string, [None])
# Initialize model
model = ACN(x, train_layers)
# link output to model output, dimension is Bx(KxD)
output = model.output

# Op for calculating the loss
with tf.name_scope("loss"):

    # get loss, dimension is batch_size
    for i in range(24):
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

# Get the batch data from dataset
traindata = np.loadtxt('traindata.txt')
train_steps_per_epoch = traindata.shape[0] / batch_size
def get_one_batch(step):
    # based on batch_size, get (batch_size / 2 * 4) images resized by 227 x 227 x 3
    for i in range(batch_size / 2 * 4):

        img = Image.open(traindata[(step - 1)*12] + i)
        img_resize = img.resize((227, 227, 3), Image.ANTIALIAS)
        img2nda = np.array(img_resize)

        if x == 0:
            x = img2nda
        else:
            x = np.concatenate((x, img2nda))

    # get one batch whose size is batch_size * 3
    for i in range(batch_size / 2):

        if i == 0:
            x_one_batch = x[i * 4]
        else:
            x_one_batch = np.concatenate((x_one_batch, x[i * 4]))

        x_one_batch = np.concatenate((x_one_batch, x[i * 4 + 1]))
        x_one_batch = np.concatenate((x_one_batch, x[i * 4 + 2]))
        x_one_batch = np.concatenate((x_one_batch, x[i * 4]))
        x_one_batch = np.concatenate((x_one_batch, x[i * 4 + 1]))
        x_one_batch = np.concatenate((x_one_batch, x[i * 4 + 3]))

    return x_one_batch

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

        step = 0

        # training
        while step < train_steps_per_epoch:

            trainbatch = get_one_batch(step + 1)

            sess.run(loss, feed_dict={x: trainbatch})

            if step % display_step == 0:
                writer.add_summary(s, epoch * train_steps_per_epoch + step)

            # store the output images vector from all training batches
            if step == 0:
                refervectors = tf.Session().run(output)
            else:
                refervectors = np.concatenate((refervectors, tf.Session().run(output)))

            step += 1

        # validate the model on the entire validation data every epoch

        print("{} Saving checkpoint of model...".format(datetime.now()))

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        # Initialize an saver for store model checkpoints
        saver = tf.train.Saver()
        save_path = saver.save(sess, checkpoint_name)

        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
