from __future__ import print_function
#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""

Train scrpit 

"""

import keras
from keras.models import Model
from keras.layers import Input, Lambda

from BaseModel import ShortCutNet
from DataGen import DataGen
from helper import init_gpu


from os import listdir
from os.path import isfile, join

from loss import identity_loss, regr_shape, regr_loss

# set gpu id with "0", "1", "2"
init_gpu("0")

# PATH
path_sharp = './data/Deblur/TextPatch/'
path_blur = './data/Deblur/TextBlur/'
model_path = './ModelSave/'
path_weights = model_path + 'ModelFinal.h5'

# Params
batch_size = 128
epochs = 21

# input image dimensions
img_rows = 65
img_cols = 65
num_of_dim = 1

kernel_crop = 24
label_rows = img_rows - kernel_crop
label_cols = img_cols - kernel_crop


def main():
    
    # Parameters
    params = {'img_rows': img_rows,
              'img_cols': img_cols,
              'num_of_dim': num_of_dim,
              'label_rows': label_rows,
              'label_cols': label_cols,
              'batch_size': batch_size,
              'kernel_crop': kernel_crop,
              'shuffle': True}
    
    name_read = [x for x in listdir(path_sharp) if isfile(join(path_sharp, x))]
    
    train_generator = DataGen(**params).Generator(name_read, path_sharp, path_blur)
    
    input_dim_blur = (img_rows, img_cols, num_of_dim)
    input_dim_sharp = (label_rows, label_cols, num_of_dim)
    
    DeblurNet = ShortCutNet().DeblurResidualNet(input_dim_blur, 6)
    #DeblurNet = ShortCutNet().DeblurSHCNet(input_dim_blur, 17)
    DeblurNet.summary()
    
    input_blur = Input(shape=(input_dim_blur))
    input_sharp = Input(shape=(input_dim_sharp))
    
    img_deblur = DeblurNet(input_blur)
    
    # Loss
    loss_regr = Lambda(regr_loss, output_shape=regr_shape)([img_deblur, input_sharp])
    
    # Model
    model = Model(inputs = [input_blur, input_sharp], outputs = loss_regr)
    
    # train
    model.compile(loss=identity_loss, optimizer=keras.optimizers.Adadelta())
    model.summary()
    
    #path_model_save = model_path + 'DeblurRes_{epoch:02d}-{loss:.2f}.hdf5'
    path_model_save = model_path + 'DeblurSHC_{epoch:02d}-{loss:.2f}.hdf5'
    check_point = keras.callbacks.ModelCheckpoint(path_model_save,
                                                monitor = 'loss',
                                                verbose = 0,
                                                save_best_only = False,
                                                save_weights_only = False,
                                                mode = 'auto',
                                                period = 1)
    callbacks = [check_point]
    
    # Train the model on the dataset
    model.fit_generator(generator = train_generator,
                        steps_per_epoch = int(len(name_read)/batch_size),
                        epochs = epochs,
                        callbacks = callbacks,
                        use_multiprocessing = True,
                        verbose = 1)
    

    model.save(path_weights)

if __name__ == '__main__':
    main()