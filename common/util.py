import tensorflow as tf
import yaml

def preprocess(image_tensor, label_tensor, config):
    # resize and cast
    image_tensor = tf.image.resize_images(image_tensor, (config.image_size, config.image_size))
    image_tensor = tf.cast(image_tensor, tf.float32)

    label_tensor = tf.image.resize_images(label_tensor, (config.image_size, config.image_size))
    if config.label_channel == 3:
        label_tensor = tf.subtract(tf.divide(tf.multiply(label_tensor, [2.0]), [255.0]), [1.0])

    # subtrac mean if need
    if config.mean:
        image_tensor = image_tensor - config.mean

    return image_tensor, label_tensor


class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)

def load_config(conf_file):
    with open(conf_file) as f:
        config = Config(**yaml.load(f))

    return config

