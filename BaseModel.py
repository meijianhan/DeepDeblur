#!/usr/bin/env python2
# -*- coding: utf-8 -*-


'''
Shortcut Network Model

'''

from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Activation, Input
from keras.layers.merge import add


class ShortCutNet():
    '''Generate networks with short cut
    '''
    def __init__(self):
        '''Initialization
        '''
        self.num_filters = 128
        self.kernel_size = (3, 3)
        self.strides = (1, 1)
        self.padding = "same"
        self.kernel_initializer = "he_normal"
        #self.kernel_regularizer = l2(1.e-4)
        self.kernel_regularizer = None
        self.trainable = True


    def _BNRelu(self, x):
        """Helper to build a BN with relu block
        """
        x = BatchNormalization()(x)
        return Activation("relu")(x)


    def _ConvOnly(self, x):
        
        return Conv2D(filters = self.num_filters,
                     kernel_size = self.kernel_size,
                     strides = self.strides,
                     padding = self.padding,
                     kernel_initializer = self.kernel_initializer,
                     kernel_regularizer = self.kernel_regularizer,
                     trainable = self.trainable)(x)


    def _ConvBNRelu(self, x):
        
        x = Conv2D(filters = self.num_filters,
                             kernel_size = self.kernel_size,
                             strides = self.strides,
                             padding = self.padding,
                             kernel_initializer = self.kernel_initializer,
                             kernel_regularizer = self.kernel_regularizer,
                             trainable = self.trainable)(x)

        return self._BNRelu(x)


    def _AddShortCut(self, x):

        x = add(x)
        return self._BNRelu(x)


    def DeblurResidualNet(self, input_shape, num_block):
        '''Base residual net network for deblur.
        '''

        input_x = Input(shape=input_shape)

        # Model
        # trainable flag just for debug
        self.trainable = True
        #self.trainable = False

        self.kernel_size = (25, 25)
        self.padding = "valid"
        layer_25by25 = self._ConvBNRelu(input_x)

        self.kernel_size = (3, 3)
        self.padding = "same"
        shortcut = self._ConvBNRelu(layer_25by25)

        for i in xrange(0, num_block):
            x = self._ConvBNRelu(shortcut)
            x = self._ConvBNRelu(x)
            x = self._ConvOnly(x)
            shortcut = self._AddShortCut([x, shortcut])

        self.num_filters = 1
        x = self._ConvOnly(shortcut)
        model = Model(inputs=input_x, outputs=x)

        return model


    def DeblurSHCNet(self, input_shape, num_block):
        '''Base SHC net network for deblur.
        '''

        input_x = Input(shape=input_shape)

        # Model
        # trainable flag just for debug
        self.trainable = True
        #self.trainable = False

        self.kernel_size = (25, 25)
        self.padding = "valid"
        layer_25by25 = self._ConvBNRelu(input_x)

        self.kernel_size = (3, 3)
        self.padding = "same"
        shortcut = self._ConvBNRelu(layer_25by25)
        x = self._ConvBNRelu(shortcut)
        for i in xrange(0, num_block):
            x = self._ConvOnly(x)
            x = self._AddShortCut([x, shortcut])

        self.num_filters = 1
        x = self._ConvOnly(x)
        model = Model(inputs=input_x, outputs=x)

        return model