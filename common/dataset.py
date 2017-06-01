from random import shuffle
from util import read_img
import cv2
import numpy as np

class BathLoader(object):
    def __init__(self, config):
        self.source = config.source
        if hasattr(config, 'val_source'):
            self.val_source = config.val_source
        else:
            self.val_source = None

        if hasattr(config, 'mean'):
            self.mean = config.mean
            assert len(self.mean) == 3
        else:
            self.mean = [0, 0, 0]

        if hasattr(config, 'mirror'):
            self.mirror = config.mirror
            assert isinstance(self.mirror, bool)
        else:
            self.mirror = False

        self.train_list = [line.rstrip('\n') for line in open(self.source)]
        self.val_list = [line.rstrip('\n') for line in open(self.val_source)] if self.val_source is not None else None
        shuffle(self.train_list)
        shuffle(self.val_list)
        self.image_size = config.image_size
        self.label_size = config.label_size
        self.batch_size = config.batch_size
        if hasattr(config, 'val_batch'):
            self.val_batch = config.val_batch
        else:
            self.val_batch = 1
        self._train_cur = 0
        self._epoch = 1
        self._val_cur = 0

    def _load_next_image(self, is_train=True):
        """
        load next image in a batch
        :param is_train:
        :return:
        """
        if is_train:
            if self._train_cur == len(self.train_list):
                self._train_cur = 0
                shuffle(self.train_list)
                print '{} epoch finished'.format(self._epoch)
                self._epoch += 1
            [image_file, label_file] = self.train_list[self._train_cur].split()
        else:
            if self._val_cur == len(self.val_list):
                self._val_cur = 0
                shuffle(self.val_list)
            [image_file, label_file] = self.val_list[self._val_cur].split()

        # read the image and label image
        img = read_img(image_file)
        label = read_img(label_file)

        # reshape if needed
        img = cv2.resize(img, self.image_size)
        label = cv2.resize(label, self.label_size)

        # random flip if needed
        if self.mirror and is_train:
            flip = np.random.choice(2) * 2 - 1
            img = img[:, ::flip, :]
            label = label[:, ::flip, :]
        # add the cur
        if is_train:
            self._train_cur += 1
        else:
            self._val_cur += 1

        # subtract the mean and convert
        label = np.float32(label)
        img = np.float32(img)
        img -= self.mean
        return img, label

    def next_train_batch(self):
        """
        load next batch data for training
        :return:
        """
        images = np.empty((self.batch_size, self.image_size, 3))
        labels = np.empty((self.batch_size, self.label_size, 1))
        for i in range(self.batch_size):
            img, label = self._load_next_image(is_train=True)
            images[i, ...] = img
            labels[i, ...] = label
        return images, labels

    def next_val_batch(self):
        """
        load next batch data for val
        :return:
        """
        if self.val_source is None:
            raise Exception('No val source')
        images = np.empty((self.val_batch, self.image_size, 3))
        labels = np.empty((self.val_batch, self.label_size, 1))
        for i in range(self.val_batch):
            img, label = self._load_next_image(is_train=False)
            images[i, ...] = img
            labels[i, ...] = label
        return images, labels



