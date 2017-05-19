import tensorflow as tf

def _get_variable(name, shape, initializer):
    var = tf.get_variable(name, shape, initializer=initializer)
    return var

def _weights_with_weight_decay(name, shape, std, wd):
    var = _get_variable(name, shape, tf.truncated_normal_initializer(stddev=std))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def conv2d(name, inputs, input_channels, output_channels, kernel, stride,
           bias_var=0.01, wd=0.0, stddev=0.01):
    # get the weight
    with tf.variable_scope(name) as scope:
        weight_shape = [kernel, kernel, input_channels, output_channels]
        weight = _weights_with_weight_decay('w_' + name, shape=weight_shape,
                                            std=stddev, wd=wd)
        # do the convolution job
        conv = tf.nn.conv2d(inputs, weight, [1, stride, stride, 1], padding='SAME')
        # if we need the biases ?
        if bias_var:
            bias_shape = [output_channels]
            bias = _get_variable('b_' + name, bias_shape, tf.constant_initializer(bias_var))
            conv = tf.nn.bias_add(conv, bias)

        return conv


def convs(name, inputs, input_channels, output_channels, phase, kernel=3, stride=1,
          bias_var=0.01, wd=0.0, stddev=0.01):
    conv = conv2d(name, inputs, input_channels, output_channels, kernel,
                  stride, bias_var, wd, stddev)
    bn = batchnorm(name+'_bn', conv, phase)
    return relu(bn)

def deconv(name, inputs, input_channels, output_channels, kernel, stride,
           bias_var=0.0, wd=0.0, stddev=0.01):
    with tf.variable_scope(name):
        # get the weight
        weight_shape = [kernel, kernel, output_channels, input_channels]
        weight = _weights_with_weight_decay('w_' + name, weight_shape, std=stddev, wd=wd)
        # get out shape
        batch, height, width, channel = inputs.get_shape().as_list()
        out_shape = tf.stack([batch, height * stride, width * stride, output_channels])
        strides = [1, stride, stride, 1]
        conv = tf.nn.conv2d_transpose(inputs, weight, out_shape, strides, padding='SAME')
        if bias_var:
            bias_shape = [output_channels]
            bias = _get_variable('b_' + name, bias_shape, tf.constant_initializer(bias_var))
            conv = tf.nn.bias_add(conv, bias)

        return conv

def batchnorm(name, inputs, phase):
    with tf.variable_scope(name):
        out = tf.contrib.layers.batch_norm(inputs,
                                           center=True,
                                           scale=True,
                                           is_training=phase)
        return out

def relu(inputs):
    return tf.nn.relu(inputs)

def max_pool_with_mask(inputs, kernel, stride, padding='SAME'):
    return tf.nn.max_pool_with_argmax(inputs, [1, kernel, kernel, 1],
                                      [1, stride, stride, 1], padding=padding)


def unpool(inputs, mask, upratio=2):
    batch, height, width, channel = inputs.get_shape().as_list()
    out_shape = tf.stack([batch, height * upratio, width * upratio, channel])

    # update according the location
    out_values = tf.zeros(tf.reduce_prod(out_shape))
    input_flatten = tf.reshape(inputs, tf.reduce_prod(tf.shape(inputs)))
    out_values = tf.scatter_update(out_values, mask, input_flatten)

    # reshape back and return
    out = tf.reshape(out_values, out_shape)
    return out


