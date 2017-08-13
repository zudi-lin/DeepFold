#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import print_function, division

import keras_gpu_memory
import sys
import numpy as np
import scipy as sp
import h5py
import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Dropout, Activation, Flatten, Input
from keras.utils import np_utils, generic_utils
from six.moves import range
from keras.models import model_from_json

path_prefix = "/home/linzudi/DeepFold/DeepFold_1D_CNN_4/"
sys.path.append(path_prefix)
from Functions_data import prep_data
Winsize = 801

test_path= path_prefix+"DeepFold_Test"
test_data, test_label = prep_data(Winsize,test_path)
test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], 1)
model=model_from_json(open(path_prefix+'DeepFold_1D_architecture.json').read())
model.load_weights(path_prefix+'DeepFold_1D_weight.h5')

Proba = model.predict(test_data, verbose=0)
accu = 0
Thr = 0.5

for k in range(test_data.shape[0]):
    if ((Proba[k,1] > Thr and test_label[k]==1) or (Proba[k,1] < Thr and test_label[k]==0)):
        accu = accu+1
vali_accuracy = float(accu)/float(test_data.shape[0])
print("Validation accuracy: "+str(vali_accuracy))
