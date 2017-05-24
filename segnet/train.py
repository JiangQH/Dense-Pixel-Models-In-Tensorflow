import os
import os.path as osp
import sys

parent_dir = os.getcwd()
path = osp.dirname(parent_dir)
sys.path.append(path)
from model import Model
from common.denseinput import DenseInput
from common.loss import compute_dot_loss
from common.util import load_config
import tensorflow as tf
import time
import numpy as np
import argparse

def solve(config):
    with tf.Graph().as_default() as g:
        # construct the data pipline
        images, labels, masks = DenseInput(config).densedata_pipelines()
        # infere the output
        predictions = Model(config).inference(images)
        # the loss
        #loss = compute_euclidean_loss(predictions, labels, masks)
        loss = compute_dot_loss(predictions, labels, masks)
        # train op, use the sgd with momentum
        global_step = tf.Variable(0, name='global_step', trainable=False)
        #learning_rate = tf.train.exponential_decay(learning_rate=config.lr,
        #                                           global_step=global_step, decay_steps=config.stepsize,
        #                                           decay_rate=config.gamma)
        #train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=config.momentum).minimize(
        #    loss, global_step=global_step
        #)
        train_op = tf.train.AdamOptimizer(learning_rate=config.lr).minimize(loss, global_step=global_step)

        # the saver to load params of pretrained model
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True

        with tf.Session(config=sess_config) as sess:
            ckpt = tf.train.get_checkpoint_state(config.model_dir)
            if not ckpt:
                print 'no pre training model, init from first...'
                sess.run(init)
            else:
                ckpt_path = ckpt.model_checkpoint_path
                saver.restore(sess, ckpt_path)
            #for var in tf.global_variables():
            #    print var
            # begin training
            #for var in tf.trainable_variables():
            #    print var
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            print 'begin training now....'
            start_time = time.time()
            local_time = time.time()
            for step in xrange(config.max_iter+1):
                _, loss_val, pre, gt = sess.run([train_op, loss, predictions, labels])
                if step % config.display == 0 or step == config.max_iter:
                    print '{}[iterations], train loss {}, time consumes {}'.format(step,
                                                                                  loss_val,
                                                                                  time.time()-local_time)
                    local_time = time.time()
                #plt.figure(1)
                #plt.imshow(np.uint8(rgb_masked[0, ...]))
                #plt.figure(2)
                #plt.imshow(np.uint8((normal_masked[0, ...]/2 + 0.5) * 255))
                #plt.figure(3)
                #plt.imshow(np.uint8(mask_num[0,...]))
                assert not np.isnan(loss_val), 'model with loss nan'

                if step != 0 and (step % config.snapshot == 0 or step == config.max_iter):
                    print 'saving snapshot...'
                    saver.save(sess, osp.join(config.model_dir, 'model.ckpt'), global_step=step)

            coord.request_stop()
            coord.join(threads)
            print 'done, total consumes time {}'.format(time.time() - start_time)

            sess.close()


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', required=True, help='path to the config file')
    args = parser.parse_args()
    config = load_config(args.conf)
    solve(config)

if __name__ == '__main__':
    tf.app.run()


