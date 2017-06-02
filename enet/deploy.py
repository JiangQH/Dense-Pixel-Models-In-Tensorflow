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

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512
BATCH_SIZE = 4
NUM_CLASSES = 20

def deploy_encoder(imgs, out_dir, model_dir):
    with tf.Graph().as_default() as g:
        images = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE,
                                                         IMAGE_HEIGHT, IMAGE_WIDTH,
                                                         3])
        out = build_encoder(images, is_training=False, num_classes=NUM_CLASSES)
        predictions = tf.argmax(out, axis=3)
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        saver = tf.train.Saver()
        with tf.Session(config=sess_config) as sess:
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if not ckpt:
                raise Exception('No pretrained model...')
            ckpt_path = ckpt.model_checkpoint_path
            saver.restore(sess, ckpt_path)
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
                    pres = predictions.eval(session=sess, feed_dict={images: in_data})
                    # save it
                    for i in range(count):
                        pre = pres[i, ...]
                        out_name = out_names[i]
                        #plt.figure(1)
                        #plt.imshow(pre)
                        scipy.misc.imsave(out_name, pre)
                    print 'forwading done with {}'.format(time.time() - start)
                    # reset data
                    out_names = []
                    count = 0
            # the remaining forward
            pres = predictions.eval(session=sess, feed_dict={images: in_data})
            for i in range(count):
                pre = pres[i, ...]
                out_name = out_names[i]
                scipy.misc.imsave(out_name, pre)

            sess.close()

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', '-i', default='/media/jqh/My Passport/Private/dataset/cityscapes/trial/rgb', help='path to input dir')
    parser.add_argument('--out_dir', '-o', default='./output', help='path to the output dir')
    parser.add_argument('--model_dir', '-m', default='./model', help='path to the model dir')
    return parser


def main(args):
    parser = build_parser()
    args = parser.parse_args()
    indir = args.in_dir
    imgs = [osp.join(indir, f) for f in os.listdir(indir)]
    deploy_encoder(imgs, args.out_dir, args.model_dir)

if __name__ == '__main__':
    tf.app.run()

