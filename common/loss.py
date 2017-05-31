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


def compute_cross_entry_with_weight(pre, gt, probs, invalid_label=None, c=1.02):
    # use the weight_probs if provided, we mask the background out
    # the background is the max label
    # compute the weight mask first
    weighing = tf.zeros_like(probs)
    for i in range(len(probs)):
        if i != invalid_label:
            tf.scatter_update(weighing, i, tf.divide(1.0, tf.log(c + probs[i])))
    # generate the whole mask
    weight_mask = tf.gather(weighing, gt)
    # compute the softmax loss
    weighted_cross_entropy = tf.multiply(weight_mask, tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt, logits=pre))
    # the final loss
    mean_cross_entropy = tf.reduce_mean(weighted_cross_entropy, name='cross_entropy')
    # add to the total loss
    tf.add_to_collection('losses', mean_cross_entropy)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def compute_accuracy(pre, gt):
    correct = tf.equal(tf.argmax(pre, axis=3), gt)
    accuracy = tf.reduce_mean(correct)
    return accuracy
