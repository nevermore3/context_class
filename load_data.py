#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-01-05 16:03:22
# @Author  : jmq (58863790@qq.com)
# @Link    : ${link}
# @Version : $Id$

import os
import sys
import jieba
import cPickle
import numpy as np
from static import Math

global_sep = '\t'
_SPACE = b'_SPACE'
_GO = b'_GO'
_EOS = b'_EOS'
_UNK = b'_UNK'
_START_VOCAB = [_SPACE, _GO, _EOS, _UNK]

# load 停用词
def stop_word(fn='./data/stop_word'):
    s_set =set([])
    with open(fn, 'rn') as _f:
        for word in _f:
            s_set.add(word.strip().decode('utf8'))
    return s_set

def chinese_tokenizer(sentence):
    return list(jieba.cut(sentence, cut_all=False))

# max_vocabulary_size : 字典长度
def get_vocab(fn='./data/osri', qdict='./dict/q_dict.pkl', qlist='./dict/q_list', sep=global_sep, tokenizer=chinese_tokenizer,max_vocabulary_size=20000):
    q_dict=dict()
    s_set = stop_word()
    with open(fn, 'rb') as f:
        for line in f:
            s = line.strip()
            s = tokenizer(s)
            for w in s:
                if w not in s_set:
                    if w not in q_dict:
                        q_dict[w] = 1
                    else:
                        q_dict[w] += 1
    dict_list = _START_VOCAB + sorted(q_dict, key=q_dict.get, reverse=True)

    if len(dict_list) > max_vocabulary_size:
            dict_list = dict_list[:max_vocabulary_size]

    q_dict = dict([(x, y) for (y, x) in enumerate(dict_list)])
    with open(qdict, 'w') as f:
        cPickle.dump(q_dict, f)
    with open(qlist, 'w') as f:
        for line in dict_list:
            f.write(line.encode('utf8')+'\n')

    return q_dict


def data_to_token_id(q_dict='./dict/q_dict.pkl',postive='./data/postive_osri', negative='./data/negative_osri', tokenizer=chinese_tokenizer):

    seed=113
    #s_set=stop_word()
    with open(q_dict, 'r') as f:
        qdict = cPickle.load(f)

    p_list=list()
    n_list=list()

    with open(postive,'r') as _f:
        for line in _f.readlines():
            words=tokenizer(line.strip())
            line = [qdict.get(w, '3') for w in words]
            line = filter(lambda x:x!='3', line)
            if line!=[]:
                p_list.append([str(x) for x in line])

    with open(negative,'r') as _f:
        for line in _f.readlines():
            words=tokenizer(line.strip())
            line = [qdict.get(w, '3') for w in words]
            line = filter(lambda x:x!='3', line)
            if line!=[]:
                n_list.append([str(x) for x in line])

    
    postive_length=len(p_list)
    negative_length=len(n_list)

    length=postive_length+negative_length

    X=p_list+n_list



    postive_static=[len(x) for x in p_list]
    X_static=[len(x) for x in X]
    a=Math()
    temp=a.mode(postive_static)
    print('postive static mode is :',temp)
    print('postive static mean is :',np.mean(X_static))

    labels= list(np.ones(postive_length))+list(np.zeros(negative_length))

    # 正负样本shuffle
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)
 
    #训练集合测试集的比例
    split_at=int(length-length/10)

    x_train=X[:split_at]
    x_test=X[split_at+1:]
    y_train=labels[:split_at]
    y_test=labels[split_at+1:]

    with open('./data/x_train', 'w') as f:
        cPickle.dump(x_train, f)
    with open('./data/y_train', 'w') as f:
        cPickle.dump(y_train, f)
    with open('./data/x_test', 'w') as f:
        cPickle.dump(x_test, f)
    with open('./data/y_test', 'w') as f:
        cPickle.dump(y_test, f)



# 单个样本-> vector
def get_sentences(sentence, q_dict='./dict/q_dict.pkl', tokenizer=chinese_tokenizer):
    #s_set=stop_word()
    with open(q_dict, 'r') as f:
        q_dict = cPickle.load(f)
    line = sentence.strip()
    words = tokenizer(line)
    #temp=list()
    #for i  in words:
    #    if i not in s_set:
    #        temp.append(i)

    line = [q_dict.get(w, '3') for w in words]
    return line



def load_data():

    with open('./data/x_train', 'r') as f:
        x_train = cPickle.load(f)

    with open('./data/x_test', 'r') as f:
        x_test = cPickle.load(f)

    with open('./data/y_train', 'r') as f:
        y_train = cPickle.load(f)

    with open('./data/y_test', 'r') as f:
        y_test = cPickle.load(f)

    return x_train, y_train,x_test, y_test



def main():
    #get_vocab()
    data_to_token_id()


if __name__ =='__main__':
    main()



