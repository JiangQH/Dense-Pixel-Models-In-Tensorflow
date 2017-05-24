import tensorflow as tf
import yaml

def preprocess(image_tensor, label_tensor, config):
    # resize and cast
    image_tensor = tf.image.resize_images(image_tensor, (config.image_width, config.image_height))
    image_tensor = tf.cast(image_tensor, tf.float32)
    # subtrac mean if need
    if config.mean:
        image_tensor = image_tensor - config.mean


    label_tensor = tf.image.resize_images(label_tensor, (config.label_width, config.label_height))
    # split into three and add and concat back to get the final mask, if it is surface normal
    if config.label_channel == 3:
        labels = tf.split(label_tensor, 3, 2)
        mask0 = tf.cast(tf.not_equal(labels[0], 127), tf.float32)
        mask1 = tf.cast(tf.not_equal(labels[1], 127), tf.float32)
        mask2 = tf.cast(tf.not_equal(labels[2], 127), tf.float32)
        adding = tf.add(mask0, mask1)
        adding = tf.add(adding, mask2)
        mask = tf.cast(tf.not_equal(adding, 0), tf.float32)
        #mask = tf.concat([mask_slice, mask_slice, mask_slice], axis=2)
        label_tensor = tf.subtract(tf.divide(tf.multiply(label_tensor, [2.0]), [255.0]), [1.0])
    else:
        mask = tf.cast(tf.not_equal(label_tensor, 0), tf.float32)

    return image_tensor, label_tensor, mask


class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)

def load_config(conf_file):
    with open(conf_file) as f:
        config = Config(**yaml.load(f))

    return config

