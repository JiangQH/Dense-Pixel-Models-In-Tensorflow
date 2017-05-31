import os
import os.path as osp
import sys
parent_dir = os.getcwd()
path = osp.dirname(parent_dir)
sys.path.append(path)

import tensorflow as tf
from model import Model
import os
import os.path as osp
import numpy as np
from PIL import Image
import scipy.misc
import time
import argparse
def deploy(in_dir, out_dir, model_dir):
    with tf.Graph().as_default() as g:
        datas = tf.placeholder(tf.float32)
        predictions = Model().inference(datas, is_training=False)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print 'restoring model prams, loading....'
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception('No pretrained model, check it...')

        # now do the feed job
        start_time = time.time()
        print 'begin forwarding...'
        in_images = os.listdir(in_dir)
        for im_name in in_images:
            print im_name
            in_image = osp.join(in_dir, im_name)
            out_image = osp.join(out_dir, im_name)
            image_data = np.asarray(Image.open(in_image), dtype=np.float32)
            pre = predictions.eval(session=sess, feed_dict={datas: image_data})
            # save it
            scipy.misc.imsave(out_image, np.uint8((pre/2 + 0.5) * 255))
        print 'handling done, time consumes {}'.format(time.time() - start_time)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', '-i', default='/home/qinhong/project/segnet_depth/data/test/mini_data', help='path to input dir')
    parser.add_argument('--out_dir', '-o', default='./output', help='path to the output dir')
    parser.add_argument('--model_dir', '-m', required=True, help='path to the model dir')
    return parser


def main(args):
    parser = build_parser()
    args = parser.parse_args()
    deploy(args.in_dir, args.out_dir, args.model_dir)

if __name__ == '__main__':
    tf.app.run()
