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
import simplejson

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

        data_loader = BathLoader(config)
        # images, labels= DenseInput(config).densedata_pipelines()
        # val_images, val_labels = DenseInput(config).densedata_pipelines(is_training=False)
        # infer the output according to the current stage, train the encoder or train them together
        if config.train_decoder:
            encode = build_encoder(images=images, is_training=True)
            out = build_decoder(encoder=encode, is_training=True, num_classes=config.num_classes)
        else:
            # only train the encoder
            out = build_encoder(images=images, is_training=True, num_classes=config.num_classes)

        # compute the loss and accuracy
        loss = compute_cross_entry_with_weight(out, labels, config.label_probs, config.invalid_label, config.c)
        accuracy = compute_accuracy(out, labels, config.invalid_label)
        # compute the accuracy
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
            print 'begin training now'
            start_time = time.time()
            local_start_time = time.time()
            train_losses = []
            val_losses = []
            accuracies = []
            for step in xrange(config.max_iter + 1):
                # construct the feed dict, fetch the data
                imgs, gts = data_loader.next_train_batch()
                #plt.figure(1)
                #plt.imshow(np.uint8(imgs[0, ...]))
                #plt.figure(2)
                #plt.imshow(np.uint8(gts[0, ...]))
                train_feed_dict = {images: imgs, labels: gts}
                _, loss_val = sess.run([train_op, loss], feed_dict=train_feed_dict)
                if step % config.display == 0 or step == config.max_iter:
                    print '{}[iterations], time consumes {}, train loss {}'.format(step,
                                                                                     time.time() - local_start_time,
                                                                                     loss_val
                                                                                    )
                    local_start_time = time.time()

                assert not np.isnan(loss_val), 'model with loss nan'
                train_losses.append(loss_val)

                if step % config.test_iter == 0 or step == config.max_iter:
                    imgs, gts = data_loader.next_val_batch()
                    val_feed_dict = {images: imgs, labels: gts}
                    accu, val_loss_val = sess.run([accuracy, loss], feed_dict=val_feed_dict)
                    accuracies.append(accu)
                    val_losses.append(val_loss_val)
                    print 'test model, {}[iterations], with loss val {}, accuracy {}'.format(step, val_loss_val, accu)

                if step != 0 and (step % config.snapshot == 0 or step == config.max_iter):
                    imgs, gts = data_loader.next_val_batch()
                    val_feed_dict = {images: imgs, labels: gts}
                    accu, val_loss_val = sess.run([accuracy, loss], feed_dict=val_feed_dict)
                    accuracies.append(accu)
                    val_losses.append(val_loss_val)
                    print 'snapshot model with loss val {}, accuracy {}'.format(val_loss_val, accu)
                    saver.save(sess, osp.join(config.model_dir, 'model.ckpt'), global_step=global_step)


            print 'done, total time comsums {}'.format(time.time() - start_time)
            with open('train_loss_log.txt', 'w') as f:
                simplejson.dump(train_losses, f)
                f.close()
            with open('val_loss_log.txt', 'w') as f:
                simplejson.dump(val_losses, f)
                f.close()
            with open('val_accu_log.txt', 'w') as f:
                simplejson.dump(accuracies, f)
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