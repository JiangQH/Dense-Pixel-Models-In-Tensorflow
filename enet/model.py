from common.layers import conv2d, batchnorm, prelu, dilated_conv, max_pool, spatial_dropout
import tensorflow as tf



def _prelu_bn(name, inputs, is_training=True, alpha_init=0.0, decay=0.1):
    """
    combine the batch norm and prelu together
    :param name: 
    :param inputs: 
    :param is_training: 
    :param alpha_init: 
    :return: 
    """
    bn = batchnorm(name+'_bn', inputs, is_training, decay)
    return prelu(name+'_prelu', bn, alpha_init)



def _bottleneck_encoder(name, inputs, input_channels, output_channels, internal_scale=4, asy=0, dilated=0,
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
        main_branch = conv2d(name+'_main_unit1', inputs, input_channels, internal_channels, kernel_size, kernel_size,
                      bias_var=None, wd=0)
        # the prelu_bn unit
        main_branch = _prelu_bn(name+'_main_unit1', main_branch, is_training, decay=bn_decay)

        # the conv unit according to the type
        if not asy and not dilated:
            main_branch = conv2d(name+'_main_unit2', main_branch, internal_channels, internal_channels, 3, 1,
                                 bias_var=None, wd=0)
        elif asy:
            main_branch = conv2d(name+'_main_unit21', main_branch, internal_channels, internal_channels, [1, asy],
                                 1, bias_var=None, wd=None)
            main_branch = conv2d(name+'_main_unit22', main_branch, internal_channels, internal_channels, [asy, 1],
                                 1, bias_var=None, wd=None)
        elif dilated:
            main_branch = dilated_conv(name+'_main_unit2', main_branch, internal_channels, output_channels, 3,
                                       dilated, bias_var=None, wd=None)
        else:
            raise Exception("Error for bottleneck {}".format(name))
        main_branch = _prelu_bn(name+'_main_unit2', main_branch, is_training, decay=bn_decay)

        # the 1 x 1 to recover the ori channel num
        main_branch = conv2d(name+'_main_unit3', main_branch, internal_channels, output_channels, 1, 1,
                             bias_var=None, wd=0)
        main_branch = batchnorm(name+'_main_unit3_bn', main_branch, is_training, decay=bn_decay)
        # the regularizar, spatial dropout
        main_branch = spatial_dropout(main_branch, dropout_ratio, is_training)

        # the other branch can be maxpooling and padding or nothing
        other = inputs
        if downsample:
            other = max_pool(other, 2, 2)
            # zero padding to match the main branch, use concat to do the zero paddings to channel
            batches, height, width, channels = other.get_shape().as_list()
            padding_channels = output_channels - channels
            padding = tf.zeros([batches, height, width, padding_channels], dtype=tf.float32)
            # padding to the other
            other = tf.concat([other, padding], axis=3)
        # add the main_branch and other branch together
        out = tf.add(main_branch, other)
        # after a prelu init, return
        return prelu(name+'_out', out)


def _bottleneck_decoder(name, inputs, input_channels, output_channels, internal_scale=4,
                        upsample=False, reverse_module=False, bn_decay=0.1):
    with tf.variable_scope(name) as scope:
        internal_channels = output_channels / 4









