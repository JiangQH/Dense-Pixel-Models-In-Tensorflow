import tensorflow as tf

def preprocess(image_tensor, label_tensor, config):
    # resize and cast
    image_tensor = tf.image.resize_images(image_tensor, (config.image_size, config.image_size))
    image_tensor = tf.cast(image_tensor, tf.float32)

    label_tensor = tf.image.resize_images(label_tensor, (config.image_size, config.image_size))
    if config.label_channel == 3:
        label_tensor = tf.subtract(tf.divide(tf.multiply(label_tensor, [2.0]), [255.0]), [1.0])

    # subtrac mean if need
    sutracted = image_tensor - config.mean

    return image_tensor, label_tensor