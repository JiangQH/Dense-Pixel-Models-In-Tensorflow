import os
import os.path as osp
import sys
parent_dir = os.getcwd()
path = osp.dirname(parent_dir)
sys.path.append(path)

from common.layers import conv_block, max_pool_with_mask, unpool, conv2d, max_pool, norm
import numpy as np
import tensorflow as tf
class Model(object):

    def __init__(self, config):
        """
        use to load the pretrained vgg wraper
        :param model_path: model path to the pre trained vgg 16 model
        """
        self.data_dict = np.load(config.vgg_path, encoding='latin1').item()
        self.config = config

    def inference(self, images, is_training):

        # the encoder structure as vgg16, loading the pre-trained model we should
        conv1_1 = conv_block('conv1_1', is_training, images, 3, 64, bias_var=self.data_dict['conv1_1'][1],
                             wd=self.config.weight_decay,
                             weight_initializer=tf.constant_initializer(self.data_dict['conv1_1'][0]))

        conv1_2 = conv_block('conv1_2', is_training, conv1_1, 64, 64, bias_var=self.data_dict['conv1_2'][1],
                             wd=self.config.weight_decay,
                            weight_initializer=tf.constant_initializer(self.data_dict['conv1_2'][0]))

        # pool1 = max_pool(conv1_2, 2, 2)
        pool1, mask1 = max_pool_with_mask(conv1_2, 2, 2)

        conv2_1 = conv_block('conv2_1', is_training, pool1, 64, 128,  bias_var=self.data_dict['conv2_1'][1],
                        weight_initializer=tf.constant_initializer(self.data_dict['conv2_1'][0]), wd=self.config.weight_decay)

        conv2_2 = conv_block('conv2_2', is_training, conv2_1, 128, 128, bias_var=self.data_dict['conv2_2'][1],
                        weight_initializer=tf.constant_initializer(self.data_dict['conv2_2'][0]), wd=self.config.weight_decay)

        # pool2 = max_pool(conv2_2, 2, 2)
        pool2, mask2 = max_pool_with_mask(conv2_2, 2, 2)

        conv3_1 = conv_block('conv3_1', is_training, pool2, 128, 256, bias_var=self.data_dict['conv3_1'][1],
                        weight_initializer=tf.constant_initializer(self.data_dict['conv3_1'][0]), wd=self.config.weight_decay)

        conv3_2 = conv_block('conv3_2', is_training, conv3_1, 256, 256,   bias_var=self.data_dict['conv3_2'][1],
                        weight_initializer=tf.constant_initializer(self.data_dict['conv3_2'][0]), wd=self.config.weight_decay)

        conv3_3 = conv_block('conv3_3', is_training, conv3_2, 256, 256,  bias_var=self.data_dict['conv3_3'][1],
                        weight_initializer=tf.constant_initializer(self.data_dict['conv3_3'][0]), wd=self.config.weight_decay)

        # pool3 = max_pool(conv3_3, 2, 2)
        pool3, mask3 = max_pool_with_mask(conv3_3, 2, 2)

        conv4_1 = conv_block('conv4_1', is_training, pool3, 256, 512, bias_var=self.data_dict['conv4_1'][1],
                        weight_initializer=tf.constant_initializer(self.data_dict['conv4_1'][0]), wd=self.config.weight_decay)

        conv4_2 = conv_block('conv4_2', is_training, conv4_1, 512, 512, bias_var=self.data_dict['conv4_2'][1],
                        weight_initializer=tf.constant_initializer(self.data_dict['conv4_2'][0]), wd=self.config.weight_decay)

        conv4_3 = conv_block('conv4_3', is_training, conv4_2, 512, 512, bias_var=self.data_dict['conv4_3'][1],
                        weight_initializer=tf.constant_initializer(self.data_dict['conv4_3'][0]), wd=self.config.weight_decay)

        # pool4 = max_pool(conv4_3, 2, 2)
        pool4, mask4 = max_pool_with_mask(conv4_3, 2, 2)

        conv5_1 = conv_block('conv5_1', is_training, pool4, 512, 512,  bias_var=self.data_dict['conv5_1'][1],
                        weight_initializer=tf.constant_initializer(self.data_dict['conv5_1'][0]), wd=self.config.weight_decay)

        conv5_2 = conv_block('conv5_2', is_training, conv5_1, 512, 512, bias_var=self.data_dict['conv5_2'][1],
                        weight_initializer=tf.constant_initializer(self.data_dict['conv5_2'][0]), wd=self.config.weight_decay)

        conv5_3 = conv_block('conv5_3', is_training, conv5_2, 512, 512,  bias_var=self.data_dict['conv5_3'][1],
                        weight_initializer=tf.constant_initializer(self.data_dict['conv5_3'][0]), wd=self.config.weight_decay)
        # pool5 = max_pool(conv5_3, 2, 2)
        pool5, mask5 = max_pool_with_mask(conv5_3, 2, 2)



        # the decoder part, unpool and conv
        up5 = unpool(pool5, mask5)
        # up5 = unpool_cpu(pool5)
        up5c = conv_block('up5c', is_training, up5, 512, 512,  bias_var=0.1, wd=self.config.weight_decay)
        up5b = conv_block('up5b', is_training, up5c, 512, 512, bias_var=0.1, wd=self.config.weight_decay)
        up5a = conv_block('up5a', is_training, up5b, 512, 512,  bias_var=0.1, wd=self.config.weight_decay)

        up4 = unpool(up5a, mask4)
        # up4 = unpool_cpu(up5a)
        up4c = conv_block('up4c', is_training, up4, 512, 512,  bias_var=0.1, wd=self.config.weight_decay)
        up4b = conv_block('up4b', is_training, up4c, 512, 512,  bias_var=0.1, wd=self.config.weight_decay)
        up4a = conv_block('up4a', is_training, up4b, 512, 256,  bias_var=0.1, wd=self.config.weight_decay)

        up3 = unpool(up4a, mask3)
        # up3 = unpool_cpu(up4a)
        up3c = conv_block('up3c', is_training, up3, 256, 256,  bias_var=0.1, wd=self.config.weight_decay)
        up3b = conv_block('up3b', is_training, up3c, 256, 256,  bias_var=0.1, wd=self.config.weight_decay)
        up3a = conv_block('up3a', is_training, up3b, 256, 128,  bias_var=0.1, wd=self.config.weight_decay)

        up2 = unpool(up3a, mask2)
        # up2 = unpool_cpu(up3a)
        up2b = conv_block('up2b', is_training, up2, 128, 128, bias_var=0.1, wd=self.config.weight_decay)
        up2a = conv_block('up2a', is_training, up2b, 128, 64,  bias_var=0.1, wd=self.config.weight_decay)

        up1 = unpool(up2a, mask1)
        # up1 = unpool_cpu(up2a)
        up1b = conv_block('up1', is_training, up1, 64, 64, bias_var=0.1, wd=self.config.weight_decay)
        out = conv2d(name='out', inputs=up1b, input_channels=64,
                     output_channels=self.config.num_classes, kernel=3, stride=1, bias_var=0.1, wd=self.config.weight_decay)
        #out_norm = norm(out)
        return out

