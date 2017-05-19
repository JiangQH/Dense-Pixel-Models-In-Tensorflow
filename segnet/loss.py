import tensorflow as tf


def compute_euclidean_loss(pre, gt, invalid_mask):
    # reshape
    batch, height, width, channel = pre.get_shape().as_list()
    pre_flat = tf.reshape(pre, [batch*height*width*channel])
    gt_flat = tf.reshape(gt, [batch*height*width*channel])
    invalid_mask_flat = tf.reshape(invalid_mask, [batch*height*width*channel])
    invalid_mask_flat = tf.cast(invalid_mask_flat, tf.float32)

    predict = tf.multiply(pre_flat, invalid_mask_flat)
    target = tf.multiply(gt_flat, invalid_mask_flat)
    squared_d = tf.square(tf.subtract(predict, target))
    total_count = tf.divide(tf.subtract(tf.cast(tf.size(gt), tf.float32), tf.reduce_sum(invalid_mask_flat)), channel)

    # get the loss including the weight decay loss
    loss = tf.divide(tf.reduce_sum(squared_d), total_count)
    tf.add_to_collection('losses', loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')



