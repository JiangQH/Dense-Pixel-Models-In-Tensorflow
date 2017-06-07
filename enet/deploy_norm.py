import os
import os.path as osp
import sys
parent_dir = os.getcwd()
path = osp.dirname(parent_dir)
sys.path.append(path)

from model import build_encoder, build_decoder
import time
import tensorflow as tf
import numpy as np
from PIL import Image
import scipy.misc
import argparse
import matplotlib.pyplot as plt
from common.util import colorize_cityscape

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 16
NUM_CLASSES = 3

def deploy(imgs, out_dir, model_path, use_decoder=False):
    with tf.Graph().as_default() as g:
        images = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE,
                                                         IMAGE_HEIGHT, IMAGE_WIDTH,
                                                         3])
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        if use_decoder:
            encode = build_encoder(images=images, is_training=is_training)
            out = build_decoder(encoder=encode, is_training=is_training, num_classes=NUM_CLASSES)
        else:
            out = build_encoder(images=images, is_training=is_training, num_classes=NUM_CLASSES)

        #predictions = tf.argmax(out, axis=3)
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        saver = tf.train.Saver()
        with tf.Session(config=sess_config) as sess:
            saver.restore(sess, model_path)
            in_data = np.zeros((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.float32)
            count = 0
            out_names = []
            for img in imgs:
                basename = osp.basename(img)
                out_name = osp.join(out_dir, basename)
                print basename
                image_data = scipy.misc.imresize(Image.open(img), (IMAGE_HEIGHT, IMAGE_WIDTH))
                image_data = np.asarray(image_data, dtype=np.float32)
                in_data[count, ...] = image_data
                out_names.append(out_name)
                count += 1
                # should we forward ?
                if count == BATCH_SIZE:
                    start = time.time()
                    pres = out.eval(session=sess, feed_dict={images: in_data, is_training: False})
                    # save it
                    #for i in range(count):
                    #    pre = pres[i, ...]
                        #pre = colorize_cityscape(pre)
                    #    out_name = out_names[i]
                    #    pre = np.uint8((pre/2.0 + 0.5) * 255)
                        #plt.figure(1)
                        #plt.imshow(pre)
                    #    scipy.misc.imsave(out_name, pre)
                    print 'forwading done with {}'.format(time.time() - start)
                    # reset data
                    out_names = []
                    count = 0
            # the remaining forward
            pres = out.eval(session=sess, feed_dict={images: in_data, is_training:False})
            for i in range(count):
                pre = pres[i, ...]
                pre = np.uint8((pre / 2.0 + 0.5) * 255)
                #pre = colorize_cityscape(pre)
                out_name = out_names[i]
                scipy.misc.imsave(out_name, pre)

            sess.close()

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', '-i', default='/home/qinhong/project/dataset/cityscape/rgb/test/bonn/', help='path to input dir')
    parser.add_argument('--out_dir', '-o', default='./output', help='path to the output dir')
    parser.add_argument('--model_path', '-m', required=True, help='path to the model dir')
    return parser


def main(args):
    parser = build_parser()
    args = parser.parse_args()
    indir = args.in_dir
    imgs = [osp.join(indir, f) for f in os.listdir(indir)]
    #lines = [line.rstrip('\n') for line in open('../data/trial_cityscape.csv', 'r')]
    #imgs = [line.split(',')[0] for line in lines]
    if not osp.exists(args.out_dir):
        os.mkdir(args.out_dir)
    deploy(imgs, args.out_dir, args.model_path, use_decoder=True)

if __name__ == '__main__':
    tf.app.run()

