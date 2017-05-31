import os
import os.path as osp
import sys
parent_dir = os.getcwd()
path = osp.dirname(parent_dir)
sys.path.append(path)

from model import build_encoder, build_decoder
from common.loss import compute_cross_entry_with_weight
from common.util import load_config
from common.denseinput import DenseInput
import tensorflow as tf
import time
import numpy as np
import argparse


def solve(config):
    with tf.Graph().as_default() as g:
        # get the data pipline
        images, labels= DenseInput(config).densedata_pipelines()
        # val_images, val_labels = DenseInput(config).densedata_pipelines(is_training=False)
        # infer the output according to the current stage, train the encoder or train them together
        if config.train_decoder:
            encode = build_encoder(images=images, is_training=True)
            out = build_decoder(encoder=encode, is_training=True, label_channel=config.label_channel)
        else:
            # only train the encoder
            out = build_encoder(images=images, is_training=True, label_channel=config.label_channel)

        # compute the loss and accuracy
        loss = compute_cross_entry_with_weight(out, labels, config.label_probs, config.c)
        # val_loss = compute_cross_entry()
        # the train op
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = tf.train.AdamOptimizer(learning_rate=config.lr).minimize(loss, global_step=global_step)

        # the saver
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            ckpt = tf.train.get_checkpoint_state(config.model_dir)
            sess.run(init)
            if not ckpt:
                print 'no pre trained model, train all from scratch...'
            else:
                ckpt_path = ckpt.model_checkpoint_path
                saver.restore(sess, ckpt_path)

            # begin the training job
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            print 'begin training now'
            start_time = time.time()
            local_start_time = time.time()
            for step in xrange(config.max_iter + 1):
                _, loss_val = sess.run([train_op, loss])
                if step % config.display == 0 or step == config.max_iter:
                    print '{}[iterations], train loss {}, time consumes {}'.format(step,
                                                                                   loss_val,
                                                                                   time.time()-local_start_time)
                    local_start_time = time.time()

                assert not np.isnan(loss_val), 'model with loss nan'

                if step != 0 and (step % config.snapshot == 0 or step == config.max_iter):
                    print 'snapshot the model'
                    saver.save(sess, osp.join(config.model_dir, 'model.ckpt'), global_step=global_step)

            coord.request_stop()
            coord.join(threads)
            print 'done, total time comsums {}'.format(time.time() - start_time)

            sess.close()


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', required=True, help='path to the config file')
    args = parser.parse_args()
    config = load_config(args.conf)
    solve(config)

if __name__ == '__main__':
    tf.app.run()