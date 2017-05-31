import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.framework import ops

import numbers

def _get_weights_stddev(shape, type='xavier'):
    """
    get the weight stddev by the kernel shape
    :param shape:
    :param type:
    :return:
    """
    if len(shape) == 1:
        fan_in = fan_out = 1
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        receiptive = 1
        for dim in shape[:-2]:
            receiptive *= dim
        fan_in = receiptive * shape[-2]
        fan_out = receiptive * shape[-1]
    scale = 1.0 / tf.maximum(1.0, (fan_in + fan_out) / 2.0)
    if type == 'xavier':
        stddev = tf.sqrt(scale)
    else:
        raise Exception('unknown weights init type {}'.format(type))
    return stddev

def _get_variable(name, shape, initializer):
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    return var

def _weights_with_weight_decay(name, shape, wd, w_initializer):
    # first get the weights stddev
    """
    if isinstance(initializer, basestring):
        stddev = _get_weights_stddev(shape, type=initializer)
    elif isinstance(initializer, numbers.Number):
        stddev = initializer
    else:
        raise Exception('weight initializer must be str or number')
    """
    if w_initializer == 'xavier':
        stddev = _get_weights_stddev(shape, w_initializer)
        w_initializer = tf.truncated_normal_initializer(stddev=stddev)

    var = _get_variable(name, shape, initializer=w_initializer)
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_decay_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv2d(name, inputs, input_channels, output_channels, kernel, stride,
           bias_var=None, wd=0.001, weight_initializer='xavier'):
    # get the weight
    with tf.variable_scope(name) as scope:
        if type(kernel) is list:
            weight_shape = kernel + [input_channels, output_channels]
        else:
            weight_shape = [kernel, kernel, input_channels, output_channels]
        weight = _weights_with_weight_decay('w_' + name, shape=weight_shape,
                                            wd=wd, w_initializer=weight_initializer)
        # do the convolution job
        conv = tf.nn.conv2d(inputs, weight, [1, stride, stride, 1], padding='SAME')
        # if we need the biases ?
        if bias_var is not None:
            bias_shape = [output_channels]
            bias = _get_variable('b_' + name, bias_shape, tf.constant_initializer(bias_var))
            conv = tf.nn.bias_add(conv, bias)

        return conv

def dilated_conv(name, inputs, input_channels, output_channels, kernel, dilated_rate,
           bias_var=None, wd=0.001, weight_initializer='xavier'):
    with tf.variable_scope(name) as scope:
        if type(kernel) is list:
            weight_shape = kernel + [input_channels, output_channels]
        else:
            weight_shape = [kernel, kernel, input_channels, output_channels]
        weight = _weights_with_weight_decay('w_'+name, weight_shape, wd, weight_initializer)

        # do the dilation job
        conv = tf.nn.atrous_conv2d(inputs, weight, rate=dilated_rate, padding='SAME')
        if bias_var is not None:
            bias_shape = [output_channels]
            bias = _get_variable('b_' + name, bias_shape, tf.constant_initializer(bias_var))
            conv = tf.nn.bias_add(conv, bias)
        return conv

def spatial_dropout(inputs, dropout_rate, is_training):
    """
    perform the spatial dropout to the whole feature map
    dropout ration is only performed on the channel space
    :param inputs:[batchs, height, width, channel]
    :param dropout_rate: dropout ratio to the channel
    :param is_training: do different job when is_training or not
    :return:
    """
    # get a noise mask to do the dropout job, get the mask shape
    if is_training:
        channel_num = inputs.get_shape().as_list()[-1]
        bernoulli_mask = tf.ceil(tf.subtract(tf.random_uniform([channel_num]), dropout_rate))
        # mask it to the whole channel feature map
        output = tf.multiply(inputs, bernoulli_mask)
        return output
    else:
        return tf.multiply(inputs, 1-dropout_rate)



def convs(name, inputs, input_channels, output_channels, phase, kernel=3, stride=1,
          bias_var=None, wd=0.001, weight_initializer='xavier'):
    conv = conv2d(name, inputs, input_channels, output_channels, kernel,
                  stride, bias_var, wd, weight_initializer)
    bn = batchnorm(name+'_bn', conv, phase)
    return relu(bn)

def deconv(name, inputs, input_channels, output_channels, kernel, stride,
           bias_var=None, wd=0.001, weight_initializer='xavier'):
    with tf.variable_scope(name):
        # get the weight
        weight_shape = [kernel, kernel, output_channels, input_channels]
        weight = _weights_with_weight_decay('w_' + name, weight_shape, wd=wd, w_initializer=weight_initializer)
        # get out shape
        batch, height, width, channel = inputs.get_shape().as_list()
        out_shape = tf.stack([batch, height * stride, width * stride, output_channels])
        strides = [1, stride, stride, 1]
        conv = tf.nn.conv2d_transpose(inputs, weight, out_shape, strides, padding='SAME')
        if bias_var is not None:
            bias_shape = [output_channels]
            bias = _get_variable('b_' + name, bias_shape, tf.constant_initializer(bias_var))
            conv = tf.nn.bias_add(conv, bias)
        return conv

def batchnorm(name, inputs, is_training=True, decay=0.9):
    with tf.variable_scope(name):
        out = tf.contrib.layers.batch_norm(inputs,
                                           center=True,
                                           scale=True,
                                           is_training=is_training,
                                           decay=decay)
        return out

def relu(inputs):
    return tf.nn.relu(inputs)

def max_pool_with_mask(inputs, kernel, stride, padding='SAME'):
    return tf.nn.max_pool_with_argmax(inputs, [1, kernel, kernel, 1],
                                      [1, stride, stride, 1], padding=padding)


def max_pool(inputs, kernel, stride, padding='SAME'):
    return tf.nn.max_pool(inputs, [1, kernel, kernel, 1],
                          [1, stride, stride, 1], padding=padding)

def norm(inputs):
    """
    norm the inputs to [-1, 1] by channel
    :param inputs:
    :return:
    """
    shapes = inputs.get_shape().as_list()
    x, y, z = tf.split(inputs, 3, axis=len(shapes)-1)
    square_sum = tf.add(tf.square(x), tf.square(y))
    square_sum = tf.add(square_sum, tf.square(z))
    eps = 1e-9
    x_norm = tf.divide(x, tf.sqrt(tf.add(square_sum, eps)))
    y_norm = tf.divide(y, tf.sqrt(tf.add(square_sum, eps)))
    z_norm = tf.divide(z, tf.sqrt(tf.add(square_sum, eps)))
    return tf.concat([x_norm, y_norm, z_norm], axis=len(shapes)-1)

def unpool_without_mask(inputs):
    shape = inputs.get_shape().as_list()
    dim = len(shape[1:-1])
    out = (tf.reshape(inputs, [-1] + shape[-dim:]))
    for i in range(dim, 0, -1):
        out = tf.concat(values=[out, tf.zeros_like(out)], axis=i)
    out_size = [-1] + [s * 2 for s in shape[1:-1] + [shape[-1] / 2]]
    out = tf.reshape(out, out_size)
    return out


def unpool(inputs, mask, upratio=2):
    batch, height, width, channel = inputs.get_shape().as_list()
    out_shape = (batch, height*upratio, width*upratio, channel)

    ones = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(batch, dtype=tf.int64), shape=[batch, 1, 1, 1])
    b = ones * batch_range
    y = mask // (out_shape[2] * out_shape[3])
    x = mask % (out_shape[2] * out_shape[3]) // out_shape[3]
    c = tf.range(channel, dtype=tf.int64)

    f = ones * c

    input_size = tf.size(inputs)
    indexs = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, input_size]))
    values = tf.reshape(inputs, [input_size])
    out = tf.scatter_nd(indexs, values, out_shape)
    return out



@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolGradWithArgmax(op, grad, unused_argmax_grad):
    return gen_nn_ops._max_pool_grad_with_argmax(op.inputs[0],
                                                 grad,
                                                 op.outputs[1],
                                                 op.get_attr("ksize"),
                                                 op.get_attr("strides"),
                                                 padding=op.get_attr("padding"))


def prelu(name, inputs, alpha_init=0.0):
    with tf.variable_scope(name) as scope:
        alpha = _get_variable('alpha', inputs.get_shape()[-1], initializer=tf.constant_initializer(alpha_init))

        return relu(inputs) + tf.multiply(alpha, (inputs - tf.abs(inputs))) * 0.5




