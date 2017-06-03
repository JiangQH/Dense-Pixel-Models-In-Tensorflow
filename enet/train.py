import os
import os.path as osp
import sys
parent_dir = os.getcwd()
path = osp.dirname(parent_dir)
sys.path.append(path)

from model import build_encoder, build_decoder
from common.loss import compute_cross_entry_with_weight, compute_accuracy
from common.util import load_config
from common.dataset import BathLoader
import tensorflow as tf
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle
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
            for var in tf.all_variables():
                name = (var.name).split(':')[0]
                encoder_params[name] = var

            out = build_decoder(encoder=encode, is_training=is_training, num_classes=config.num_classes)
        else:
            # only train the encoder
            out = build_encoder(images=images, is_training=is_training, num_classes=config.num_classes)
            for var in tf.all_variables():
                name = (var.name).split(':')[0]
                encoder_params[name] = var

        # compute the loss and accuracy
        loss = compute_cross_entry_with_weight(out, labels, config.label_probs, config.invalid_label, config.c)
        accuracy = compute_accuracy(out, labels, config.invalid_label)
        # compute the accuracy
        # val_loss = compute_cross_entry()
        # the train op
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = tf.train.AdamOptimizer(learning_rate=config.lr).minimize(loss, global_step=global_step)

        #for var in tf.trainable_variables():
        #    print var
        saver = tf.train.Saver()
        init_op = tf.global_variables_initializer()
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.allow_growth = True
        with tf.Session(config=sess_config) as sess:
            sess.run(init_op)
            if config.train_decoder:
                ckpt = tf.train.get_checkpoint_state(config.model_dir)
                if not ckpt:
                    print 'no pre trained encoder model, train all from scratch...'
                else:
                    # find all params for encoder
                    ckpt_path = ckpt.model_checkpoint_path
                    encoder_saver = tf.train.Saver(encoder_params)
                    encoder_saver.restore(sess, ckpt_path)

            # begin the training job
            print 'begin training now'
            start_time = time.time()
            local_start_time = time.time()
            train_losses = []
            val_losses = []
            accuracies = []
            train_accuracies = []
            # change it to use max_epoch to stop other than the max iter
            for step in xrange(MAX_ITER + 1):
                # construct the feed dict, fetch the data
                imgs, gts = data_loader.next_train_batch()
                #plt.figure(1)
                #plt.imshow(np.uint8(imgs[0, ...]))
                #plt.figure(2)
                #plt.imshow(np.uint8(gts[0, ...]))
                train_feed_dict = {images: imgs, labels: gts, is_training: True}
                _, loss_val, train_accu = sess.run([train_op, loss, accuracy], feed_dict=train_feed_dict)
                if step % config.display == 0 or step == MAX_ITER:
                    print '{}[iterations], time consumes {}, train loss {}, train accuracy {}'.format(step,
                                                                                     time.time() - local_start_time,
                                                                                     loss_val, train_accu
                                                                                    )
                    local_start_time = time.time()

                assert not np.isnan(loss_val), 'model with loss nan'
                train_losses.append(loss_val)
                train_accuracies.append(train_accu)

                if step % config.test_iter == 0 or step == MAX_ITER:
                    print '.............testing model..............'
                    imgs, gts = data_loader.next_val_batch()
                    val_feed_dict = {images: imgs, labels: gts, is_training: False}
                    val_accu, val_loss_val = sess.run([accuracy, loss], feed_dict=val_feed_dict)
                    accuracies.append(val_accu)
                    val_losses.append(val_loss_val)
                    print '{}[iterations], val loss {}, val accuracy {}'.format(step, val_loss_val, val_accu)

                if step != 0 and (step % config.snapshot == 0 or step == MAX_ITER):
                    print '..............snapshot model.............'
                    imgs, gts = data_loader.next_val_batch()
                    val_feed_dict = {images: imgs, labels: gts, is_training: False}
                    val_accu, val_loss_val = sess.run([accuracy, loss], feed_dict=val_feed_dict)
                    accuracies.append(val_accu)
                    val_losses.append(val_loss_val)
                    print '{}[iterations], val loss{}, val accuracy {}'.format(step, val_loss_val, val_accu)
                    if config.train_decoder:
                        saver.save(sess, osp.join(config.model_dir, 'decoder_model.ckpt'), global_step=global_step)
                    else:
                        saver.save(sess, osp.join(config.model_dir, 'encoder_model.ckpt'), global_step=global_step)


                # should we stop now ? can add accuracy support later
                if data_loader.get_epoch() == config.max_epoch + 1:
                    imgs, gts = data_loader.next_val_batch()
                    val_feed_dict = {images: imgs, labels: gts, is_training: False}
                    val_accu, val_loss_val = sess.run([accuracy, loss], feed_dict=val_feed_dict)
                    accuracies.append(val_accu)
                    val_losses.append(val_loss_val)
                    print 'training done! with validation loss val {}, accuracy {}'.format(val_loss_val, val_accu)
                    if config.train_decoder:
                        saver.save(sess, osp.join(config.model_dir, 'decoder_model.ckpt'), global_step=global_step)
                    else:
                        saver.save(sess, osp.join(config.model_dir, 'encoder_model.ckpt'), global_step=global_step)
                    break

            print 'total time comsums {}'.format(time.time() - start_time)
            with open('train_loss_log.txt', 'wb') as f:
                pickle.dump(train_losses, f)
                f.close()
            with open('train_accu_log.txt', 'wb') as f:
                pickle.dump(train_accuracies, f)
                f.close()
            with open('val_loss_log.txt', 'wb') as f:
                pickle.dump(val_losses, f)
                f.close()
            with open('val_accu_log.txt', 'wb') as f:
                pickle.dump(accuracies, f)
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