import os
import os.path as osp
import sys
parent_dir = os.getcwd()
path = osp.dirname(parent_dir)
sys.path.append(path)

from model import build_encoder, build_decoder
import time
import tensorflow as tf


IMAGE_HEIGHT = 512
IMAGE_WIDTH = 256
BATCH_SIZE = 4
NUM_CLASSES = 20
def deploy_encoder(imgs, outs, model_dir):
	with tf.Graph().as_default() as g:
		images = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
		out = build_encoder(images=images, is_training=False, num_classes=NUM_CLASSES)
		# the saver
		saver = tf.train.Saver()
		sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
        	ckpt = tf.train.get_checkpoint_state(model_dir)
        	if not ckpt:
        		raise Exception('No pretrained model found in the model dir')
        	ckpt_path = ckpt.model_checkpoint_path
        	saver.restore(sess, ckpt_path)
        	