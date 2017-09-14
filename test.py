#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2016-12-18 20:27:05
# @Author  : jmq (58863790@qq.com)
# @Link    : ${link}
# @Version : $Id$
import os
import sys
import numpy as np
import jieba
from static import Math
def stop_word(fn='./model/stop_word'):
    s_set =set([])
    with open(fn, 'rn') as _f:
        for word in _f:
            s_set.add(word.strip().decode('utf8'))
    return s_set

def chinese_tokenizer(sentence):
    return list(jieba.cut(sentence, cut_all=False))


def test():
    s=u'人民就是我们的祖国，我爱你吗啊哈哈哈小学二年级？'
    s_set=stop_word()
    temp=chinese_tokenizer(s)
    t=list()
    for i in temp:
        if i  in s_set:
            print i+'----'
            t.append(i)
    o=' '.join(t)
    print o


def hello():
    a=[3,3,3,4,4,4,5,6,5,5,5,5,5]
    m=Math()
    t=m.mode(a)
    t=m.madian(a)
    print t



if __name__=='__main__':
    #test()
    hello()