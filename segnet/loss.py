import tensorflow as tf

def compute_euclidean_loss(pre, gt, mask):
    # reshape
    batch, height, width, channel = pre.get_shape().as_list()

    pre_masked = tf.multiply(pre, mask)
    #gt_masked = tf.multiply(gt, mask)
    #rgb_masked = tf.multiply(rgb, mask)

    total_loss = tf.reduce_sum(tf.square(tf.subtract(pre_masked, gt)))
    # total_count = tf.divide(tf.reduce_sum(mask_num), channel)
    total_count = tf.reduce_sum(mask)
    # get the loss including the weight decay loss
    loss = tf.divide(total_loss, total_count)
    tf.add_to_collection('losses', loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def compute_dot_loss(pre, gt, mask):
    dots = tf.reduce_sum(tf.multiply(pre, gt))
    total_count = tf.reduce_sum(mask)
    loss = tf.multiply(tf.divide(dots, total_count), -1)
    tf.add_to_collection('losses', loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')
