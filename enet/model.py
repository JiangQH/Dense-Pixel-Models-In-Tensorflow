from common.layers import conv2d, batchnorm, prelu, dilated_conv, max_pool
import tensorflow as tf
from tensorflow.contrib.keras.python.keras.layers.core import SpatialDropout2D


def _prelu_bn(name, inputs, is_training=True, alpha_init=0.0, decay=0.1):
    """
    combine the batch norm and prelu together
    :param name: 
    :param inputs: 
    :param is_training: 
    :param alpha_init: 
    :return: 
    """
    bn = batchnorm(name, inputs, is_training, decay)
    return prelu(name, bn, alpha_init)



def _bottleneck(name, inputs, input_channels, output_channels, internal_scale=4, asy=0, dilated=0,
                downsample=False, dropout_ratio=0.1, is_training=True, bn_decay=0.1):
    """
    :param name: 
    :param inputs: 
    :param output_channels: 
    :param internal_scale: 
    :param asy: 
    :param dilated: 
    :param downsample: 
    :param dropout_ratio: 
    :return: 
    """
    with tf.variable_scope(name) as scope:
        # the main branch, downsample the scale
        internal_channels = output_channels / internal_scale

        # the 1 x 1 projection or 2 x 2 if downsampleing
        kernel_size = 2 if downsample else 1
        main_branch = conv2d('main_conv1', inputs, input_channels, internal_channels, kernel_size, kernel_size,
                      bias_var=None, wd=0)
        # the prelu_bn unit
        main_branch = _prelu_bn('main_relu1', main_branch, is_training, decay=bn_decay)

        # the conv unit according to the type
        if not asy and not dilated:
            main_branch = conv2d('main_conv2', main_branch, internal_channels, internal_channels, 3, 1,
                                 bias_var=None, wd=0)
        elif asy:
            main_branch = conv2d('main_conv2_1', main_branch, internal_channels, internal_channels, [1, asy],
                                 1, bias_var=None, wd=None)
            main_branch = conv2d('main_conv2_2', main_branch, internal_channels, internal_channels, [asy, 1],
                                 1, bias_var=None, wd=None)
        elif dilated:
            main_branch = dilated_conv('main_conv2', main_branch, internal_channels, output_channels, 3,
                                       dilated, bias_var=None, wd=None)
        else:
            raise Exception("Error for bottleneck {}".format(name))
        main_branch = _prelu_bn('main_relu2', main_branch, is_training, decay=bn_decay)

        # the 1 x 1 to recover the ori channel num
        main_branch = conv2d('main_conv3', main_branch, internal_channels, output_channels, 1, 1,
                             bias_var=None, wd=0)
        main_branch = _prelu_bn('main_relu3', main_branch, is_training, decay=bn_decay)

        # the regularizar





