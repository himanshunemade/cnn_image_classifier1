import cv2
import numpy as np
import subprocess
import easygui

class DataSet:

    def __init__(self, images, labels, ids, cls):
        self._num_examples = images.shape[0]
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0/255.0)

        self._images = images
        self._labels = labels
        self._ids = ids
        self._cls = cls
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def ids(self):
        return self._ids

    @property
    def cls(self):
        return self._cls

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self,choice, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1

            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch
        if choice==2:
            file = easygui.fileopenbox()
            img = cv2.imread(file, 1)
            cv2.imshow('original image', img)
            img=cv2.resize(img, (64, 64), cv2.INTER_LINEAR)
            img = img.astype(np.float32)
            img = np.multiply(img, 1.0 / 255.0)

            cv2.imshow('preprocessed image', img)
            cv2.waitKey(0)
            self._images[start:end]=img
            if "SOB_B" in file:
                self._cls[start:end]='benign'
                self._labels[start:end]=[1,0]
            else:
                self._cls[start:end]='malignant'
                self._labels[start:end]=[0,1]
            return self._images[start:end], self._labels[start:end], self._ids[start:end], self._cls[start:end]


        return self._images[start:end], self._labels[start:end], self._ids[start:end],self._cls[start:end]
