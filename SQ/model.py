import tensorflow as tf
from common.layers import conv2d

def fire_module(name, inputs, input_channels, wd, weight_init='xavier', bias_var=0.01):
    with tf.VariableScope(name) as scope:
        # first is the a 1 * 1 convolution with 16 kernels output
        main = conv2d(name=name+'_main', inputs=inputs, input_channels=input_channels,
                      output_channels=16, kernel=1, stride=1, bias_var=bias_var, wd=wd, weight_initializer=weight_init)
        # a elu layer

