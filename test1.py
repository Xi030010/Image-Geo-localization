import tensorflow as tf

with tf.Session() as sess:
    output = tf.constant(tf.random_normal([72, 3]))
    init = tf.global_variables_initializer()
    sess.run(init)
    # oriimg = output[1 * 3]
    # oriimg_aux1 = output[1 * 3 + 1]
    # oriimg_aux2 = output[1 * 3 + 2]
    # ed_ori2aux1 = tf.sqrt(tf.reduce_sum(tf.square(oriimg - oriimg_aux1)))
    # ed_ori2aux2 = tf.sqrt(tf.reduce_sum(tf.square(oriimg - oriimg_aux2)))
    # print(ed_ori2aux1.get_shape())
    # s = tf.Variable(ed_ori2aux1 - ed_ori2aux2 + 0.25)
    # print(s.get_shape())
    # # sess.run(s)
    # loss = tf.maximum(s, 0)
    # print(loss.get_shape())
    for i in range(24):
        oriimg = output[i * 3]
        oriimg_aux1 = output[i * 3 + 1]
        oriimg_aux2 = output[i * 3 + 2]
        ed_ori2aux1 = tf.sqrt(tf.reduce_sum(tf.square(oriimg - oriimg_aux1)))
        ed_ori2aux2 = tf.sqrt(tf.reduce_sum(tf.square(oriimg - oriimg_aux2)))
        sum = tf.Variable([ed_ori2aux1 - ed_ori2aux2 + 0.25])
        if i == 0:
            loss = tf.maximum(sum, 0)
        else:
            loss = tf.concat(0, [loss, tf.maximum(sum, 0)])
        print(sess.run(loss))

    print(loss.get_shape())
    # print(sess.run(output))
    # print(sess.run(output))
    # print(loss.get_shape())
