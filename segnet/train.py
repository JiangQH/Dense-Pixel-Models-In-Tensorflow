import os
import os.path as osp
import sys

parent_dir = os.getcwd()
path = osp.dirname(parent_dir)
sys.path.append(path)
from model import Model
from common.dataset import BathLoader
from common.loss import compute_dot_loss, compute_euclidean_loss, compute_accuracy, compute_cross_entropy, compute_cross_entropy_with_weight
from common.util import load_config, uniform_normal
import tensorflow as tf
import time
import numpy as np
import argparse
import pickle
MAX_ITER = 9999999
def solve(config):
    with tf.Graph().as_default() as g:

        images = tf.placeholder(dtype=tf.float32, shape=[config.batch_size,
                                                         config.image_size[1], config.image_size[0], 3])
        if config.label_channel == 1:
            labels = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, config.label_size[1],
                                                             config.label_size[0]])
        else:
            labels = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, config.label_size[1],
                                                             config.label_size[0], 3])

        is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        data_loader = BathLoader(config)
        # infere the output
        predictions = Model(config).inference(images, is_training, num_classes=config.num_classes, weight_decay=config.weight_decay)

        # the loss, different for different task
        loss = compute_euclidean_loss(predictions, labels, invalid_label=config.invalid_label)
        #loss = compute_dot_loss(predictions, labels, invalid_label=config.invalid_label)

        #accuracy = compute_accuracy(predictions, labels, config.invalid_label)
        # train op, use the sgd with momentum
        global_step = tf.Variable(0, name='global_step', trainable=False)
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
                print '------restoring from {}'.format(ckpt_path)
                saver.restore(sess, ckpt_path)

            print 'begin training now....'
            train_losses = []
            val_losses = []
            start_time = time.time()
            local_time = time.time()
            for step in xrange(MAX_ITER+1):
                imgs, gts = data_loader.next_train_batch()
                gts = uniform_normal(gts)
                train_feed_dict = {images: imgs, labels: gts, is_training: True}
                _, loss_train = sess.run([train_op, loss], feed_dict=train_feed_dict)
                if step % config.display == 0 or step == MAX_ITER:
                    print '{}[iterations], time consumes {}, train loss {}'.format(step,
                                                                                  time.time() - local_time,
                                                                                  loss_train
                                                                                    )
                    local_time = time.time()
                assert not np.isnan(loss_train), 'model with loss nan'
                train_losses.append(loss_train)



                if hasattr(config, 'test_source') and (step % config.test_iter == 0 or step == MAX_ITER):
                    print '.............testing model..............'
                    imgs, gts = data_loader.next_val_batch()
                    gts = uniform_normal(gts)
                    val_feed_dict = {images: imgs, labels: gts, is_training: True}
                    loss_val= sess.run([loss], feed_dict=val_feed_dict)
                    val_losses.append(loss_val)
                    print '{}[iterations], val loss {}'.format(step, loss_val)

                if step != 0 and (step % config.snapshot == 0 or step == MAX_ITER):
                    print '..............snapshot model.............'
                    if hasattr(config, 'test_source'):
                        imgs, gts = data_loader.next_val_batch()
                        gts = uniform_normal(gts)
                        val_feed_dict = {images: imgs, labels: gts, is_training: True}
                        loss_val = sess.run([loss], feed_dict=val_feed_dict)
                        val_losses.append(loss_val)
                        print '{}[iterations], val loss {}'.format(step, loss_val)
                    saver.save(sess, osp.join(config.model_dir, 'segnet_model.ckpt'), global_step=global_step)

                # should we stop now ? can add accuracy support later
                if data_loader.get_epoch() == config.max_epoch + 1:
                    if hasattr(config, 'test_source'):
                        imgs, gts = data_loader.next_val_batch()
                        gts = uniform_normal(gts)
                        val_feed_dict = {images: imgs, labels: gts, is_training: True}
                        loss_val = sess.run([loss], feed_dict=val_feed_dict)
                        val_losses.append(loss_val)
                        print '{}[iterations], val loss {}'.format(step, loss_val)
                    saver.save(sess, osp.join(config.model_dir, 'segnet_model.ckpt'), global_step=global_step)
                    break

            print 'total time comsums {}'.format(time.time() - start_time)
            with open(osp.join(config.model_dir, 'train_loss_log.txt'), 'wb') as f:
                pickle.dump(train_losses, f)
                f.close()
            if hasattr(config, 'test_source'):
                with open(osp.join(config.model_dir, 'val_loss_log.txt'), 'wb') as f:
                    pickle.dump(val_losses, f)
                    f.close()
            sess.close()


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', required=True, help='path to the config file')
    args = parser.parse_args()
    config = load_config(args.conf)
    solve(config)

if __name__ == '__main__':
    tf.app.run()


