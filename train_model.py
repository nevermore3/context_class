#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-01-05 15:35:34
# @Author  : jmq (58863790@qq.com)
# @Link    : ${link}
# @Version : $Id$

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers import Convolution1D, GlobalMaxPooling1D
from keras.layers import  MaxPooling1D
from keras.models import load_model
from keras.models import model_from_json
from load_data  import load_data
'''
   依赖包： h5py、keras、tensorflow、numpy,....
'''

def train_lstm():
    # set parameters:
    max_features = 20000  #max_feature 字典长度
    maxlen = 10 # cut texts after this number of words (among top max_features most common words)
    batch_size = 32
    embedding_dims = 50  #词向量的长度

    print('Loading data...')
    X_train, y_train, X_test, y_test = load_data()
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen, value=3)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen, value=3)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    y_train=np.array(y_train) 
    y_test=np.array(y_test)
    print('y_train shape', y_train.shape)
    print('y_train shape', y_test.shape)


    print('Build model...')
    model = Sequential()

    model.add(Embedding(max_features, embedding_dims, dropout=0.2))
    model.add(LSTM(32, dropout_W=0.2, dropout_U=0.2,return_sequences=False))  # try using a GRU instead, for fun

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    print('Train...')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=2,validation_data=(X_test, y_test))
    score, acc = model.evaluate(X_test, y_test,batch_size=batch_size)

    #save the model 
    # serialize model to JSON
    model_json = model.to_json()
    with open("./dict/model_lstm.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./dict/model_lstm_weights.h5")
    print("Saved model to disk")
    #model.save('./dict/model_lstm.h5')
    del model

    print('Test score:', score)
    print('Test accuracy:', acc)



def trian_cnn():
    # set parameters:
    max_features = 20000
    maxlen = 8
    batch_size = 32
    embedding_dims = 50
    nb_filter = 250
    filter_length = 3
    hidden_dims = 250
    nb_epoch = 2
    print('Loading data...')
    X_train, y_train, X_test, y_test = load_data()
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen, value=3)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen, value=3)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    y_train=np.array(y_train)
    print('y_train shape', y_train.shape)
    y_test=np.array(y_test)
    print('y_train shape', y_test.shape)

    print('Build model...')
    model = Sequential()

    model.add(Embedding(max_features,embedding_dims,input_length=maxlen,dropout=0.2))

    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X_train, y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, y_test))

    score, acc = model.evaluate(X_test, y_test,batch_size=batch_size)


    #save the model &serialize model to JSON
    model_json = model.to_json()
    with open("./dict/model_cnn.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./dict/model_cnn_weights.h5")
    print("Saved model to disk")

    del model

    print('Test score:', score)
    print('Test accuracy:', acc)

def train_cnn_lstm():
    # Embedding
    max_features = 20000
    maxlen = 10
    embedding_size =40

    # Convolution
    filter_length = 5
    nb_filter = 64
    pool_length = 4

    # LSTM
    lstm_output_size = 70

    # Training
    batch_size = 35
    nb_epoch = 2

    '''
    Note:
    batch_size is highly sensitive.
    Only 2 epochs are needed as the dataset is very small.
    '''

    print('Loading data...')
    #(X_train, y_train), (X_test, y_test) = load_data()
    X_train, y_train, X_test, y_test = load_data()
    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    print('Pad sequences (samples x time)')
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen, value=3)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen, value=3)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    y_train=np.array(y_train)
    print('y_train shape', y_train.shape)
    y_test=np.array(y_test)
    print('y_train shape', y_test.shape)

    print('Build model...')

    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(0.25))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(X_test, y_test))
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)



    #save the model &serialize model to JSON
    model_json = model.to_json()
    with open("./dict/model_cnn_lstm.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("./dict/model_cnn_lstm_weights.h5")
    print("Saved model to disk")

    del model
    print('Test score:', score)
    print('Test accuracy:', acc)


if __name__ =='__main__':
    #train_cnn_lstm()
    train_lstm()
    #trian_cnn()




