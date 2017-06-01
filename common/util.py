import tensorflow as tf
import yaml
import Image
import numpy as np

def preprocess(image_tensor, label_tensor, config):
    # resize and cast
    image_tensor = tf.image.resize_images(image_tensor, config.image_size)
    image_tensor = tf.cast(image_tensor, tf.float32)
    # subtrac mean if need
    if config.mean:
        image_tensor = image_tensor - config.mean


    label_tensor = tf.image.resize_images(label_tensor, config.label_size)
    # split into three and add and concat back to get the final mask, if it is surface normal
    if config.label_channel == 3:
        label_tensor = tf.subtract(tf.divide(tf.multiply(label_tensor, [2.0]), [255.0]), [1.0])

    return image_tensor, label_tensor


class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)

def load_config(conf_file):
    with open(conf_file) as f:
        config = Config(**yaml.load(f))

    return config

def compute_mask(label_tensor, invalid_value=0):
    """
    channels be one or multiple. all channel equals the invalid_value means this location is invalid
    :param label_tensor:
    :param channels:
    :return:
    """

    shapes = label_tensor.get_shape().as_list()
    if len(shapes) == 3:
        mask = tf.cast(tf.not_equal(label_tensor, invalid_value), tf.float32)
    else:
        batch, height, width, channel = shapes
        summing = tf.zeros([batch, height, width], tf.float32)
        splits = tf.split(label_tensor, num_or_size_splits=channel, axis=3)
        for split in splits:
            split_mask = tf.cast(tf.not_equal(split, invalid_value), tf.float32)
            summing = tf.add(summing, split_mask)
        mask = tf.cast(tf.not_equal(summing, 0), tf.float32)
    return mask


def read_img(img_file):
    return np.asarray(Image.open(img_file))