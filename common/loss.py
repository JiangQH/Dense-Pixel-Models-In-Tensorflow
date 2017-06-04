import tensorflow as tf
from common.util import compute_mask

def compute_euclidean_loss(pre, gt):
    # reshape
    # batch, height, width, channel = pre.get_shape().as_list()
    mask = compute_mask(gt, invalid_value=127)
    # should we rescale the gt ?
    # gt = tf.multiply(tf.add(tf.divide(gt, 255.0), 0.5), 2.0)
    gt_masked = tf.multiply(gt, mask)
    pre_masked = tf.multiply(pre, mask)
    #gt_masked = tf.multiply(gt, mask)
    #rgb_masked = tf.multiply(rgb, mask)
    total_loss = tf.reduce_sum(tf.square(tf.subtract(pre_masked, gt_masked)))
    # total_count = tf.divide(tf.reduce_sum(mask_num), channel)
    total_count = tf.reduce_sum(mask)
    # get the loss including the weight decay loss
    loss = tf.divide(total_loss, total_count)
    tf.add_to_collection('losses', loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def compute_dot_loss(pre, gt):
    mask = compute_mask(gt, invalid_value=127)
    # gt = tf.multiply(tf.add(tf.divide(gt, 255.0), 0.5), 2.0)
    gt_masked = tf.multiply(gt, mask)
    dots = tf.reduce_sum(tf.multiply(pre, gt_masked))
    total_count = tf.reduce_sum(mask)
    loss = tf.multiply(tf.divide(dots, total_count), -1)
    tf.add_to_collection('losses', loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def compute_cross_entropy_with_weight(pre, gt, probs, invalid_label=None, c=1.02):
    # use the weight_probs if provided, we mask the background out
    # the background is the max label
    # compute the weight mask first
    weighing = []
    for i in range(len(probs)):
        if i != invalid_label:
            weighing.append(tf.divide(1.0, tf.log(c + probs[i])))
    weighing = tf.stack(weighing)
    # generate the whole mask
    gt = tf.cast(gt, tf.int64)
    weight_mask = tf.gather(weighing, gt)
    # compute the softmax loss
    weighted_cross_entropy = tf.multiply(weight_mask, tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt, logits=pre))
    # the final loss
    mean_cross_entropy = tf.reduce_mean(weighted_cross_entropy, name='cross_entropy')
    # add to the total loss
    tf.add_to_collection('losses', mean_cross_entropy)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def compute_cross_entropy(pre, gt, invalid_label=None):
    mask = compute_mask(gt, invalid_value=invalid_label)
    valid_cross_entropy = tf.multiply(mask, tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt, logits=pre))
    mean_cross_entropy = tf.reduce_mean(valid_cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', mean_cross_entropy)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def compute_accuracy(pre, gt, invalid_label=None):
    gt = tf.cast(gt, tf.int64)
    correct = tf.cast(tf.equal(tf.argmax(pre, axis=3), gt), tf.float32)
    #correct = tf.cast(tf.equal(gt, gt), tf.float32)
    # get the mask
    if invalid_label is not None:
        mask = compute_mask(gt, invalid_value=19)
        # mask the invalid out
        correct = tf.multiply(mask, correct)
        total_val = tf.reduce_sum(correct)
        total_count = tf.reduce_sum(mask)
        accuracy = tf.divide(total_val, total_count)
    else:
        accuracy = tf.reduce_mean(correct)
    return accuracy
