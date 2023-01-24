import tensorflow as tf


def ResBlock(x, num):
    with tf.variable_scope("res_block_" + str(num), reuse=tf.AUTO_REUSE):
        tmp = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], padding="same", strides=[1, 1],
                               activation=tf.nn.relu)
        tmp = tf.layers.conv2d(inputs=tmp, filters=64, kernel_size=[3, 3], padding="same", strides=[1, 1],
                               activation=tf.nn.relu)
    return x + tmp


def ResBlocks(x, num):
    with tf.variable_scope("res_blocks_" + str(num), reuse=tf.AUTO_REUSE):
        tmp = ResBlock(x, 0)
        tmp = ResBlock(tmp, 1)
        tmp = ResBlock(tmp, 2)
    return x+tmp


def mv_refine_net(x):
    x1 = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], padding="same", strides=[1, 1],
                           activation=tf.nn.relu)
    x2 = tf.layers.conv2d(inputs=x1, filters=64, kernel_size=[3, 3], padding="same", strides=[2, 2],
                           activation=tf.nn.relu)
    x3 = tf.layers.conv2d(inputs=x2, filters=64, kernel_size=[3, 3], padding="same", strides=[2, 2],
                          activation=tf.nn.relu)
    x1_r = ResBlocks(x1, 0)
    x2_r = ResBlocks(x2, 1)
    x3_r = ResBlocks(x3, 2)
    _, h2, w2, _ = tf.unstack(tf.shape(x2))
    x3_up = tf.image.resize_bilinear(x3_r, (h2, w2))
    _, h1, w1, _ = tf.unstack(tf.shape(x1))
    x2_up = tf.image.resize_bilinear((x3_up+x2_r), (h1, w1))
    x4 = tf.layers.conv2d(inputs=(x1_r+x2_up), filters=2, kernel_size=[3, 3], padding="same", strides=[1, 1],
                           activation=tf.nn.relu)
    output = x4 + x

    return output


def rf_refine_net(x):
    x1 = tf.layers.conv2d(inputs=x, filters=64, kernel_size=[3, 3], padding="same", strides=[1, 1],
                           activation=tf.nn.relu)
    x2 = tf.layers.conv2d(inputs=x1, filters=64, kernel_size=[3, 3], padding="same", strides=[2, 2],
                           activation=tf.nn.relu)
    x3 = tf.layers.conv2d(inputs=x2, filters=64, kernel_size=[3, 3], padding="same", strides=[2, 2],
                          activation=tf.nn.relu)
    x1_r = ResBlocks(x1, 0)
    x2_r = ResBlocks(x2, 1)
    x3_r = ResBlocks(x3, 2)
    _, h2, w2, _ = tf.unstack(tf.shape(x2))
    x3_up = tf.image.resize_bilinear(x3_r, (h2, w2))
    _, h1, w1, _ = tf.unstack(tf.shape(x1))
    x2_up = tf.image.resize_bilinear((x3_up+x2_r), (h1, w1))
    x4 = tf.layers.conv2d(inputs=(x1_r+x2_up), filters=3, kernel_size=[3, 3], padding="same", strides=[1, 1],
                           activation=tf.nn.relu)
    output = x4 + x

    return output

