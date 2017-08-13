#!/usr/bin/env python

'''
    Deepfold-1D
    Predicting the one-dimensional structure of RNA sequences.
'''
from __future__ import absolute_import
from __future__ import print_function

##################
# Load Functions #
##################

import keras_gpu_memory
import sys
import numpy as np
import scipy as sp
import h5py
import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Dropout, Activation, Flatten, Input
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
from keras.layers.noise import GaussianNoise
from keras.optimizers import SGD
from keras import regularizers
from keras.utils import np_utils, generic_utils
from six.moves import range

##################
# Initialization #
##################

path_prefix = "/home/linzudi/DeepFold/DeepFold_1D/"
sys.path.append(path_prefix)
from Functions_data import prep_data
Winsize = 801

dir_path = path_prefix+"DeepFold_Train"
test_path= path_prefix+"DeepFold_Validate"
training_data,  label = prep_data(Winsize, dir_path)
test_data, test_label = prep_data(Winsize,test_path)
label = np_utils.to_categorical(label, 2)
# test_label = np_utils.to_categorical(test_label, 2)
print(training_data.shape[0], ' training samples')

# data_format="channels_last"
training_data = training_data.reshape(training_data.shape[0], training_data.shape[1], training_data.shape[2], 1)
test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], 1)
print("The shape of the input is: ", training_data.shape)

#################
# Construct DNN #
#################

L2_regularizers = 3e-3
def cnn_block(input, nb_kernels):
    conv1 = Conv2D(nb_kernels, kernel_size=(1,3), strides=(1,1), padding="valid", activation="relu",
                   kernel_regularizer=regularizers.l2(L2_regularizers))(input)
    norm1 = BatchNormalization(momentum=0.99, epsilon=0.001, center=True)(conv1)
    pool1 = MaxPooling2D(pool_size=(1,2), padding='same')(norm1)
    print("pool1 ", pool1._keras_shape)
    return pool1

nb_kernels = 32
input = Input(shape=(6, Winsize, 1))
print("input ", input._keras_shape)

noise = GaussianNoise(0.3)(input)
conv1 = Conv2D(nb_kernels, kernel_size=(6,9), strides=(1,1), padding="valid", activation="relu",
               kernel_regularizer=regularizers.l2(L2_regularizers))(noise)
norm1 = BatchNormalization(momentum=0.99, epsilon=0.001, center=True)(conv1)
pool1 = MaxPooling2D(pool_size=(1,2), padding='same')(norm1)
print("pool1 ", pool1._keras_shape)

act1 = cnn_block(pool1,nb_kernels)
act2 = cnn_block(act1, nb_kernels*2)
act3 = cnn_block(act2, nb_kernels*2)
act4 = cnn_block(act3, nb_kernels*4)
act5 = cnn_block(act4, nb_kernels*4)

temp1 = Flatten()(act5)
temp2 = Dropout(0.25)(temp1)
output= Dense(2, kernel_regularizer=regularizers.l2(L2_regularizers))(temp2)
output = Activation('softmax')(output)
model = Model(inputs=input, outputs=output)
model.summary()

#SGD + momentum
#model.compile provides the cross entropy loss function
sgd = SGD(lr=0.001, decay=1e-7, momentum=0.95, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
          
#########################
# Training & Validation #
#########################

#Fit the model:
print('Begin training the model...')
best_vali_accuracy = 0.
Thr = 0.5

for epoch in xrange(2000):
    print("Number of epoch: ", epoch)
    hist = model.fit(training_data, label, batch_size=32, epochs=1, shuffle=True, verbose=1)
    Proba = model.predict(test_data, verbose=0)
    accu = 0
    for k in range(test_data.shape[0]):
        if ((Proba[k,1] > Thr and test_label[k]==1) or (Proba[k,1] < Thr and test_label[k]==0)):
            accu = accu+1
    vali_accuracy = float(accu)/float(test_data.shape[0])
    if vali_accuracy > best_vali_accuracy:
        print("Accuracy is improved!")
        best_vali_accuracy = vali_accuracy
        json_string = model.to_json()
        open(path_prefix+'DeepFold_1D_architecture.json','w').write(json_string)
        model.save_weights(path_prefix+'DeepFold_1D_weight.h5', overwrite=True)
