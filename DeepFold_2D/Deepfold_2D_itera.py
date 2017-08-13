#!/usr/bin/env python

'''
    Deepfold-2D
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

##################
# Load Functions #
##################

#import keras_gpu_memory
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
Winsize = 801

#################
# Construct DNN #
#################

L2_regularizers = 5e-5

#----------------

def residual_block(input, nb_kernels, doPooling=False):
    
    conv1 = Conv2D(nb_kernels, kernel_size=(1,3), strides=(1,1), padding="same", activation="relu",
                   kernel_regularizer=regularizers.l2(L2_regularizers))(input)
    norm1 = BatchNormalization(momentum=0.99, epsilon=0.001, center=True)(conv1)
    conv2 = Conv2D(nb_kernels, kernel_size=(1,3), strides=(1,1), padding="same", activation="relu",
                   kernel_regularizer=regularizers.l2(L2_regularizers))(norm1)
    norm2 = BatchNormalization(momentum=0.99, epsilon=0.001, center=True)(conv2)
    
    # If maxpooling is conducted, the pure identity mapping need to be changed to a linear mapping
    if (doPooling):
        pool  = MaxPooling2D(pool_size=(1,2), padding='same')(norm2)
        conv3 = Conv2D(nb_kernels, kernel_size=(1,3), strides=(1,1), padding="same", activation=None)(pool)
        input = Conv2D(nb_kernels, kernel_size=(1,1), strides=(1,2), padding="same", activation=None)(input)
        added = keras.layers.add([conv3, input])
        
    else:
        pool = norm2
        conv3 = Conv2D(nb_kernels, kernel_size=(1,3), strides=(1,1), padding="same", activation=None)(pool)
        added = keras.layers.add([conv3, input])
        
    activ = Activation('relu')(added)
    print("The shape of the output ", activ._keras_shape)
    return activ

#----------------

nb_kernels = 24
input = Input(shape=(9, Winsize, 1))
print("input ", input._keras_shape)

noise = GaussianNoise(0.25)(input)
conv1 = Conv2D(nb_kernels, kernel_size=(9,11), strides=(1,1), padding="valid", activation="relu",
               kernel_regularizer=regularizers.l2(L2_regularizers))(noise)
norm1 = BatchNormalization(momentum=0.99, epsilon=0.001, center=True)(conv1)
pool1 = MaxPooling2D(pool_size=(1,2), padding='same')(norm1)
print("pool1 ", pool1._keras_shape)

block1 = residual_block(pool1, nb_kernels, doPooling=False)
block2 = residual_block(block1,nb_kernels, doPooling=False)

nb_kernels = 2*nb_kernels
block3 = residual_block(block2,nb_kernels, doPooling=True)
block4 = residual_block(block3,nb_kernels, doPooling=False)
block5 = residual_block(block4,nb_kernels, doPooling=False)

nb_kernels = 2*nb_kernels
block6 = residual_block(block5,nb_kernels, doPooling=True)
block7 = residual_block(block6,nb_kernels, doPooling=False)
block8 = residual_block(block7,nb_kernels, doPooling=False)

nb_kernels = 2*nb_kernels
block9  = residual_block(block8, nb_kernels, doPooling=True)
block10 = residual_block(block9, nb_kernels, doPooling=False)
block11 = residual_block(block10,nb_kernels, doPooling=False)

nb_kernels = 2*nb_kernels
block12 = residual_block(block11,nb_kernels, doPooling=True)
block13 = residual_block(block12,nb_kernels, doPooling=False)
block14 = residual_block(block13,nb_kernels, doPooling=False)

g_avep = GlobalAveragePooling2D()(block14)
g_avep = Dropout(0.3)(g_avep)
output = Dense(2, kernel_initializer='normal', kernel_regularizer=regularizers.l2(L2_regularizers))(g_avep)
output = Activation('softmax')(output)

model = Model(inputs=input, outputs=output)

#SGD + momentum
#model.compile provides the cross entropy loss function
sgd = SGD(lr=0.01, decay=1e-7, momentum=0.95, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

##################
# Initialization #
##################

from Iterative_2D import prep_data_2D, prep_data_2D_iterative, load_test
path_prefix = "/home/linzudi/DeepFold_2D/DeepFold_2D_RES_1/"
sys.path.append(path_prefix)
dir_path = path_prefix+"DeepFold_Train"
test_path= path_prefix+"DeepFold_Validate"

No_pair, Pair, Data= prep_data_2D(Winsize, dir_path)
ratio = int(len(No_pair)/len(Pair))
print("Number of positive samples: "+str(len(Pair)))
print("Ratio: "+str(ratio))
test_No_pair, test_Pair, test_Data = prep_data_2D(Winsize, test_path)

#Fit the model:
print('Begin training the model...')
best_vali_accuracy = 0.
Thr = 0.5
X_pos, y_pos, X_neg, y_neg = load_test(Winsize, test_No_pair, test_Pair, test_Data)
X_pos = X_pos.reshape(X_pos.shape[0], X_pos.shape[1], X_pos.shape[2], 1)
X_neg = X_neg.reshape(X_neg.shape[0], X_neg.shape[1], X_neg.shape[2], 1)
fl = open(path_prefix+"accuracy.txt","w")

for index in range(100):
    for j in range(ratio):
       	training_data, label = prep_data_2D_iterative(Winsize, j, No_pair, Pair, Data)
        training_data = training_data.reshape(training_data.shape[0], training_data.shape[1], training_data.shape[2], 1)
        label = np_utils.to_categorical(label, 2)
        #Fit the model
        print('Begin training the model...Round: '+str(index*ratio+j)+", big epoch: "+str(index))
        hist = model.fit(training_data, label, batch_size=32, epochs=1, shuffle=True, verbose=1)
        del(training_data)
        del(label)
        fl.write(str(hist.history)+"\n")

        Proba_pos = model.predict(X_pos, verbose=0)
        accu_pos = 0
        for k in range(X_pos.shape[0]):
            if (Proba_pos[k,1] > Thr):
                accu_pos = accu_pos+1
        vali_accu_pos = float(accu_pos)/float(X_pos.shape[0])
        print("vali_accu_pos: ",vali_accu_pos)

        Proba_neg = model.predict(X_neg, verbose=0)
        accu_neg = 0
        for k in range(X_neg.shape[0]):
            if (Proba_neg[k,1] < Thr):
                accu_neg = accu_neg+1
        vali_accu_neg = float(accu_neg)/float(X_neg.shape[0])
        print("vali_accu_neg: ",vali_accu_neg)

        accuracy = vali_accu_pos
        fl.write(str(best_vali_accuracy)+"\t"+str(vali_accu_pos)+"\t"+str(vali_accu_neg)+"\n")
        if (accuracy > best_vali_accuracy and vali_accu_neg > 0.99):
            best_vali_accuracy = accuracy
            json_string = model.to_json()
            open(path_prefix+'DeepFold_2D_architecture.json','w').write(json_string)
            model.save_weights(path_prefix+'DeepFold_2D_weight.h5', overwrite=True)
