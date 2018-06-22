#!/usr/bin/env python2
# -*- coding: utf-8 -*-


"""
Data Generator

"""

import numpy as np
import cv2


class DataGen():
    '''Generates data for Keras
    '''
    def __init__(self, img_rows, img_cols, num_of_dim, label_rows, label_cols, batch_size, kernel_crop, shuffle):
        '''Initialization
        '''
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.num_of_dim = num_of_dim
        self.label_rows = label_rows
        self.label_cols = label_cols
        self.batch_size = batch_size
        self.kernel_crop = kernel_crop/2
        self.shuffle = shuffle

    def Generator(self, name_read, path_sharp, path_blur):
        'Generates batches of samples'
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            idx = self._GetExplorationOrder(name_read)
            idx_len = len(idx)

            # Generate batches
            idx_max = int(idx_len/self.batch_size)
            for idx_i in range(idx_max):
                # Find list of IDs
                tmp = [name_read[i] for i in idx[idx_i*self.batch_size:(idx_i + 1)*self.batch_size]]
                # Generate data
                x, y = self._GenerateBatch(tmp, path_sharp, path_blur)
                yield x, y

            idx_i = idx_i + 1
            if((idx_len - (idx_i*self.batch_size)) > 0):
                tmp = [name_read[i] for i in idx[idx_i*self.batch_size:idx_len]]
                # Generate data
                x, y = self._GenerateBatch(tmp, path_sharp, path_blur)
                yield x, y

    def _GetExplorationOrder(self, name_read):
        'Generates order of exploration'
        # Find exploration order
        idx = np.arange(len(name_read), dtype = int)
        if self.shuffle == True:
            np.random.shuffle(idx)

        return idx

    def _GenerateBatch(self, tmp, path_sharp, path_blur):
        '''Generates data of batch_size samples
        '''
        # Initialization
        x_blur = np.zeros([self.batch_size, self.img_rows, self.img_cols, self.num_of_dim], dtype = np.float64)
        y_sharp = np.zeros([self.batch_size, self.label_rows, self.label_cols, self.num_of_dim], dtype = np.float64)
        y_fake = np.zeros([self.batch_size], dtype = int)

        # Generate data
        for count_i, name_i in enumerate(tmp):

            # Read blurry input images
            x = cv2.imread(path_blur + name_i, cv2.IMREAD_GRAYSCALE)
            x = x.reshape(self.img_rows, self.img_cols, self.num_of_dim)
            x_blur[count_i, :] = x/255.0

            # Read sharp labels
            x = cv2.imread(path_sharp + name_i, cv2.IMREAD_GRAYSCALE)
            x = x.reshape(self.img_rows, self.img_cols, self.num_of_dim)
            x = x[self.kernel_crop:(self.img_rows - self.kernel_crop), \
                    self.kernel_crop:(self.img_cols - self.kernel_crop)]
            y_sharp[count_i, :] = x/255.0
            

        return [x_blur, y_sharp], y_fake