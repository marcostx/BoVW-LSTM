"""

//[]------------------------------------------------------------------------[]
//|                                                                          |
//|                      Long-Short Term Memory Module                       |
//|                               Version 1.0                                |
//|                                                                          |
//|              Copyright 2016-2020, Marcos Vinicius Teixeira               |
//|                          All Rights Reserved.                            |
//|                                                                          |
//[]------------------------------------------------------------------------[]
//
OVERVIEW: lstm.py
//  ================================
//  This module implement a Simple Long-Short Term Memory(LSTM), a Recurrent 
//  Neural Network to sequence data and it's an adaptation of the lstm code 
//  provided by pybrain for sequence classification.
//
"""

from __future__ import print_function
import numpy as np
#np.set_printoptions(threshold=np.nan)
np.random.seed(1337)  # for reproducibility
import common
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.datasets import imdb

max_features = 20000
maxlen = 100  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(X_train, y_train) = common.generate_ucf_dataset('UCF-101/')

print(len(X_train), 'train sequences')

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
print('X_train shape:', X_train.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(LSTM(128))  # try using a GRU instead, for fun
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              class_mode="categorical")

print("Train...")
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=3,show_accuracy=True)
score, acc = model.evaluate(X_train, y_train,
                            batch_size=batch_size,
                            show_accuracy=True)
predicted = model.predict(X_train)
print('Test score:', score)
print('Test accuracy:', acc)
print('prediction = ', predicted)
print('y =',y_train[40])

