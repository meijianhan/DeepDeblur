from __future__ import print_function
#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Loss functions

"""

from keras import backend as K

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - y_true, axis=0)
    #return (y_pred - y_true)


def regr_shape(x):
    y, _ = x
    return (y[0], 1)


def regr_loss(x):
    y1, y2 = x

    l = K.sum(K.sum(K.square(y1 - y2), axis=1), axis = 1)

    return l