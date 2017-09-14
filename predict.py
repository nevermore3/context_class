#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-01-06 13:11:14
# @Author  : jmq (58863790@qq.com)
# @Link    : ${link}
# @Version : $Id$

import os
import sys
from keras.preprocessing import sequence
from keras.models import model_from_json
from keras.models import load_model
import numpy as np
from load_data import get_sentences

def predict(sentence, model_name='./dict/model_lstm.json', model_weight='./dict/model_lstm_weights.h5'):

    #set parameters:
    maxlen=20
    # load json and create model
    json_file = open(model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(model_weight)
    print("Loaded model from disk")

    x=get_sentences(sentence.decode('utf8'))
    x=np.array(x)
    #如果批量预测的话 输入需要更改为 [ [sample],[sample],[sample].....【sample] 格式应该是 ：numpy list 格式
    x=x.reshape(1,len(x))
    x=sequence.pad_sequences(x ,maxlen=maxlen, value=3)

    print x
    #predict_y=model.predict(x,batch_size=1)
    predict_y=model.predict_classes(x, batch_size=1)
    #predict_y=model.predict_proba(x,batch_size=1)
    print predict_y


if __name__ =='__main__':
    s="请问您在哪里上课？"
    #s="电话是1232131"
    predict(s)
