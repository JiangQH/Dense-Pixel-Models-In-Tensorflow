import tensorflow as tf
from common.util import compute_mask

def compute_euclidean_loss(pre, gt, invalid_label=0.0):

    mask = compute_mask(gt, invalid_value=invalid_label)
    squared = tf.square(tf.subtract(pre, gt))
    total_loss = tf.reduce_sum(tf.multiply(mask, squared))
    total_count = tf.reduce_sum(mask)
    # get the loss including the weight decay loss
    loss = tf.divide(total_loss, total_count)
    tf.add_to_collection('losses', loss)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def compute_dot_loss(pre, gt, invalid_label=0.0):
    mask = compute_mask(gt, invalid_value=invalid_label)
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
    ori_mask = compute_mask(gt, invalid_value=invalid_label)
    masked_gt = tf.cast(tf.multiply(gt, ori_mask), tf.int64)
    weighing = []
    for i in range(len(probs)+1):
        if i != invalid_label:
            weighing.append(tf.divide(1.0, tf.log(c + probs[i])))
        else:
            weighing.append(tf.constant(0.0))
    weighing = tf.stack(weighing)
    # generate the whole mask
    gt = tf.cast(gt, tf.int64)
    weight_mask = tf.gather(weighing, gt)
    # compute the softmax loss
    weighted_cross_entropy = tf.multiply(weight_mask, tf.nn.sparse_softmax_cross_entropy_with_logits(labels=masked_gt, logits=pre))
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
    correct = tf.cast(tf.equal(tf.argmax(pre, axis=3), tf.cast(gt, tf.int64)), tf.float32)
    #correct = tf.cast(tf.equal(gt, gt), tf.float32)
    # get the mask
    if invalid_label is not None:
        mask = compute_mask(gt, invalid_value=19)
        #gt = tf.cast(gt, tf.int64)
        # mask the invalid out
        correct = tf.multiply(mask, correct)
        total_val = tf.reduce_sum(correct)
        total_count = tf.reduce_sum(mask)
        accuracy = tf.divide(total_val, total_count)
    else:
        accuracy = tf.reduce_mean(correct)
    return accuracy
