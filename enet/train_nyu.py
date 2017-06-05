import os
import os.path as osp
import sys
parent_dir = os.getcwd()
path = osp.dirname(parent_dir)
sys.path.append(path)

from model import build_encoder, build_decoder
from common.loss import compute_cross_entropy_with_weight, compute_accuracy, compute_dot_loss, compute_euclidean_loss
from common.util import load_config, colorize_cityscape, uniform_normal
from common.dataset import BathLoader
import tensorflow as tf
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle
import scipy.misc
MAX_ITER = 9999999

def solve(config):
    with tf.Graph().as_default() as g:
        # get the data pipline
        images = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, config.image_size[1],
                                                         config.image_size[0], 3])
        # if label channel is one we ignore the label channel
        if config.label_channel == 1:
            labels = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, config.label_size[1],
                                                             config.label_size[0]])
        else:
            labels = tf.placeholder(dtype=tf.float32, shape=[config.batch_size, config.label_size[1],
                                                             config.label_size[0], 3])

        is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        data_loader = BathLoader(config)
        # images, labels= DenseInput(config).densedata_pipelines()
        # val_images, val_labels = DenseInput(config).densedata_pipelines(is_training=False)
        # infer the output according to the current stage, train the encoder or train them together
        encoder_params = {}
        if config.train_decoder:
            encode = build_encoder(images=images, is_training=is_training)
            # load the encoder params
            for var in tf.global_variables():
                name = (var.name).split(':')[0]
                encoder_params[name] = var

            out = build_decoder(encoder=encode, is_training=is_training, num_classes=config.num_classes)
        else:
            # only train the encoder
            out = build_encoder(images=images, is_training=is_training, num_classes=config.num_classes)
            for var in tf.global_variables():
                name = (var.name).split(':')[0]
                encoder_params[name] = var

        # compute the loss and accuracy
        #loss = compute_cross_entropy_with_weight(out, labels, config.label_probs, config.invalid_label, config.c)
        #accuracy = compute_accuracy(out, labels, config.invalid_label)
        loss = compute_euclidean_loss(out, labels, config.invalid_label)
        # compute the accuracy
        # val_loss = compute_cross_entry()
        # the train op
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = tf.train.AdamOptimizer(learning_rate=config.lr).minimize(loss, global_step=global_step)

        #for var in tf.trainable_variables():
        #    print var
        saver = tf.train.Saver(max_to_keep=None)
        init_op = tf.global_variables_initializer()
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            sess.run(init_op)
            ckpt = tf.train.get_checkpoint_state(config.model_dir)
            if not ckpt:
                print 'no pre trained encoder model, train all from scratch...'
            else:
                # find all params for encoder
                ckpt_path = ckpt.model_checkpoint_path
                print '----loading encoder params form {}'.format(ckpt_path)
                encoder_saver = tf.train.Saver(encoder_params)
                encoder_saver.restore(sess, ckpt_path)

            # begin the training job
            print 'begin training now'
            start_time = time.time()
            local_start_time = time.time()
            train_losses = []
            val_losses = []

            for step in xrange(MAX_ITER + 1):
                # construct the feed dict, fetch the data
                imgs, gts = data_loader.next_train_batch()
                # rescale the gts
                gts = uniform_normal(gts)
                train_feed_dict = {images: imgs, labels: gts, is_training: True}
                _, loss_train, preds = sess.run([train_op, loss, out], feed_dict=train_feed_dict)

                if step % config.display == 0 or step == MAX_ITER:
                    print '{}[iterations], time consumes {}, train loss {}'.format(step,
                                                                                     time.time() - local_start_time,
                                                                                     loss_train
                                                                                    )
                    local_start_time = time.time()

                assert not np.isnan(loss_train), 'model with loss nan'
                train_losses.append(loss_train)


                if hasattr(config, 'test_source') and (step % config.test_iter == 0 or step == MAX_ITER):
                    print '.............testing model..............'
                    imgs, gts = data_loader.next_val_batch()
                    gts = uniform_normal(gts)
                    val_feed_dict = {images: imgs, labels: gts, is_training: True}
                    val_loss_val = sess.run([loss], feed_dict=val_feed_dict)

                    val_losses.append(val_loss_val)
                    print '{}[iterations], val loss {}'.format(step, val_loss_val)

                if step != 0 and (step % config.snapshot == 0 or step == MAX_ITER):
                    print '..............snapshot model.............'
                    if hasattr(config, 'test_source'):
                        imgs, gts = data_loader.next_val_batch()
                        gts = uniform_normal(gts)
                        val_feed_dict = {images: imgs, labels: gts, is_training: True}
                        val_loss_val = sess.run([loss], feed_dict=val_feed_dict)

                        val_losses.append(val_loss_val)
                        print '{}[iterations], val loss {}'.format(step, val_loss_val)
                    if config.train_decoder:
                        saver.save(sess, osp.join(config.model_dir, 'decoder_model.ckpt'), global_step=global_step)
                    else:
                        saver.save(sess, osp.join(config.model_dir, 'encoder_model.ckpt'), global_step=global_step)


                # should we stop now ? can add accuracy support later
                if data_loader.get_epoch() == config.max_epoch + 1 :
                    if hasattr(config, 'test_source'):
                        imgs, gts = data_loader.next_val_batch()
                        gts = uniform_normal(gts)
                        val_feed_dict = {images: imgs, labels: gts, is_training: True}
                        val_loss_val = sess.run([loss], feed_dict=val_feed_dict)
                        # val_loss_val = sess.run([loss], feed_dict=val_feed_dict)
                        # accuracies.append(val_accu)
                        val_losses.append(val_loss_val)
                        print '{}[iterations], val loss {}'.format(step, val_loss_val)
                    if config.train_decoder:
                        saver.save(sess, osp.join(config.model_dir, 'decoder_model.ckpt'), global_step=global_step)
                    else:
                        saver.save(sess, osp.join(config.model_dir, 'encoder_model.ckpt'), global_step=global_step)
                    break


            print 'total time comsums {}'.format(time.time() - start_time)
            with open('train_loss_nyu.txt', 'wb') as f:
                pickle.dump(train_losses, f)
                f.close()
            if hasattr(config, 'test_source'):
                with open('val_loss_nyu.txt', 'wb') as f:
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