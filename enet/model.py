from common.layers import conv2d, batchnorm, prelu, dilated_conv, max_pool, spatial_dropout, relu, deconv, unpool_without_mask
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
                downsample=False, dropout_ratio=0.1, is_training=True, bn_decay=0.1, wd=2e-4):
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
                      bias_var=None, wd=wd)
        # the prelu_bn unit
        main_branch = _prelu_bn(name+'_main_unit1', main_branch, is_training, decay=bn_decay)

        # the conv unit according to the type
        if not asy and not dilated:
            main_branch = conv2d(name+'_main_unit2', main_branch, internal_channels, internal_channels, 3, 1,
                                 bias_var=None, wd=wd)
        elif asy:
            main_branch = conv2d(name+'_main_unit21', main_branch, internal_channels, internal_channels, [1, asy],
                                 1, bias_var=None, wd=wd)
            main_branch = conv2d(name+'_main_unit22', main_branch, internal_channels, internal_channels, [asy, 1],
                                 1, bias_var=None, wd=wd)
        elif dilated:
            main_branch = dilated_conv(name+'_main_unit2', main_branch, internal_channels, internal_channels, 3,
                                       dilated, bias_var=None, wd=wd)
        else:
            raise Exception("Error for bottleneck {}".format(name))
        main_branch = _prelu_bn(name+'_main_unit2', main_branch, is_training, decay=bn_decay)

        # the 1 x 1 to recover the ori channel num
        main_branch = conv2d(name+'_main_unit3', main_branch, internal_channels, output_channels, 1, 1,
                             bias_var=None, wd=wd)
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
                        upsample=False, reverse_module=False, is_training=True, bn_decay=0.1, wd=2e-4):
    with tf.variable_scope(name) as scope:
        internal_channels = output_channels / internal_scale

        # the main branch
        main_branch = conv2d(name+'_main_unit1', inputs, input_channels, internal_channels,
                             1, 1, bias_var=None, wd=wd)
        main_branch = batchnorm(name+'_main_unit1_bn', main_branch, is_training,
                                decay=bn_decay)
        main_branch = relu(main_branch)

        # the second conv, decide by upsample or not
        if upsample:
            main_branch = deconv(name+'_main_unit2', main_branch, internal_channels,
                                 internal_channels, 3, 2, bias_var=None, wd=wd)
        else:
            main_branch = conv2d(name+'_main_unit2', main_branch, internal_channels,
                                 internal_channels, 3, 1, bias_var=None, wd=wd)
        main_branch = batchnorm(name+'_main_unit2_bn', main_branch, is_training,
                                decay=bn_decay)
        main_branch = relu(main_branch)
        # the third branch
        main_branch = conv2d(name+'_main_unit3', main_branch, internal_channels, output_channels,
                             1, 1, bias_var=None, wd=wd)

        # the other branch
        other = inputs
        if input_channels != output_channels or upsample:
            other = conv2d(name+'_other_unit1', other, input_channels,
                           output_channels, 1, 1, bias_var=None, wd=wd)
            other = batchnorm(name+'_other_unit1_bn', other, is_training, decay=bn_decay)
            if upsample and reverse_module:
                other = unpool_without_mask(other)

        if not upsample or reverse_module:
            main_branch = batchnorm(name+'_main_unit3_bn', main_branch, is_training, decay=bn_decay)
        else:
            return main_branch

        out = tf.add(main_branch, other)
        return relu(out)

def _initial_block(name, inputs, input_channels=3, output_channel=13, kerne=3, stride=2, wd=2e-4):
    conv = conv2d(name+'_conv_unit', inputs, input_channels, output_channel,
                  kerne, stride, bias_var=None, wd=wd)
    pool = max_pool(inputs, 2, 2)
    out = tf.concat([conv, pool], axis=3)
    return out


def build_encoder(images, is_training=True, label_channel=None):
    # the init block
    encode = _initial_block('initial', images)
    # the bottleneck 1.0
    encode = _bottleneck_encoder('bottleneck1.0', encode, 16, 64,
                                 downsample=True, dropout_ratio=0.01, is_training=is_training)
    # the bottleneck 1.1-1.4
    for i in range(4):
        encode = _bottleneck_encoder('bottleneck1.{}'.format(i+1), encode, 64, 64,
                                     dropout_ratio=0.01, is_training=is_training)
    # the bottleneck2.0
    encode = _bottleneck_encoder('bottleneck2.0', encode, 64, 128,
                                 downsample=True, is_training=is_training)

    # the bottleneck2.1-2.8, 3.1-3.8
    for i in range(2):
        encode = _bottleneck_encoder('bottleneck{}.1'.format(i+2), encode, 128, 128, is_training=is_training)
        encode = _bottleneck_encoder('bottleneck{}.2'.format(i+2), encode, 128, 128, dilated=2, is_training=is_training)
        encode = _bottleneck_encoder('bottleneck{}.3'.format(i+2), encode, 128, 128, asy=5, is_training=is_training)
        encode = _bottleneck_encoder('bottleneck{}.4'.format(i+2), encode, 128, 128, dilated=4, is_training=is_training)
        encode = _bottleneck_encoder('bottleneck{}.5'.format(i+2), encode, 128, 128, is_training=is_training)
        encode = _bottleneck_encoder('bottleneck{}.6'.format(i+2), encode, 128, 128, dilated=8, is_training=is_training)
        encode = _bottleneck_encoder('bottleneck{}.7'.format(i+2), encode, 128, 128, asy=5, is_training=is_training)
        encode = _bottleneck_encoder('bottleneck{}.8'.format(i+2), encode, 128, 128, dilated=16, is_training=is_training)

    if label_channel is not None:
        # train the encoder first
        encode = conv2d('prediction', encode, 128, label_channel, 1, 1, bias_var=None, wd=0)

    return encode


def build_decoder(encoder, is_training=True, label_channel=3):
    # upsamle model
    decode = _bottleneck_decoder('bottleneck4.0', encoder, 128, 64, upsample=True, reverse_module=True,
                                 is_training=is_training)
    decode = _bottleneck_decoder('bottleneck4.1', decode, 64, 64, is_training=is_training)
    decode = _bottleneck_decoder('bottleneck4.2', decode, 64, 64, is_training=is_training)
    decode = _bottleneck_decoder('bottleneck5.0', decode, 64, 16, upsample=True, reverse_module=True,
                                 is_training=is_training)
    decode = _bottleneck_decoder('bottleneck5.1', decode, 16, 16, is_training=is_training)
    # the output
    out = deconv('prediction', decode, 16, label_channel, 2, 2, bias_var=None, wd=0)

    return out









