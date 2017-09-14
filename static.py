#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-01-06 17:41:39
# @Author  : jmq (58863790@qq.com)
# @Link    : ${link}
# @Version : $Id$

import os

class Math(object):
    """docstring for Math"""
    #def __init__(self, arg):
    #    super(Math, self).__init__()
    #    self.arg = arg

    #求中位数
    def madian(self,t):
        t=sorted(t)
        if len(t)%2==1:
            return t[len(t)/2]
        else:
            return (t[len(t)/2-1]+ t[len(t)/2])/2.0
    # 求众数
    def mode(self,l):
        count_dict={}
        for i in l:
            if count_dict.has_key(i):
                count_dict[i]+=1
            else:
                count_dict[i]=1
        max_appear=0
        for v in count_dict.values():
            if v>max_appear:
                max_appear=v

        mode_list=[]
        for k,v in count_dict.items():
            if v==max_appear:
                mode_list.append(k)

        return mode_list



