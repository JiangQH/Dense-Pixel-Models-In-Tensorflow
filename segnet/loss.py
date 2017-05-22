import tensorflow as tf

def compute_euclidean_loss(pre, gt, mask):
    # reshape
    batch, height, width, channel = pre.get_shape().as_list()

    mask_num = tf.cast(mask, tf.float32)
    pre_masked = tf.multiply(pre, mask_num)
    gt_masked = tf.multiply(gt, mask_num)

    total_loss = tf.reduce_sum(tf.square(tf.subtract(pre_masked, gt_masked)))
    total_count = tf.divide(tf.reduce_sum(mask_num), channel)
    # get the loss including the weight decay loss
    loss = tf.divide(total_loss, total_count)
    tf.add_to_collection('losses', loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')



