#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import lmdb
import sys
sys.path.insert(0,'/usr/local/moji/caffe/python')
import caffe
import os
import cv2
import sys
import numpy as np
from glob import glob as gl

class_num = 30
train_image_path = '/mnt/sdc/zihao.chen/cloudRecognize/data/train/image'
def read_data():
    train_data_list=[]
    train_lable_list = []
    test_data_list=[]
    test_lable_list = []
    for class_index in range(class_num):
        all_pics_path = gl(os.path.join(train_image_path, '%d_*.png' % class_index))
        print len(all_pics_path)
        nn = range(len(all_pics_path))
        np.random.seed(23)
        np.random.shuffle(nn)
        all_pics_path = all_pics_path[nn]
        train_data_list +=all_pics_path[:500]
        test_data_list +=all_pics_path[500:]
        train_lable_list+=[class_index for i in range(len(all_pics_path[:500]))]
        test_lable_list += [class_index for i in range(len(all_pics_path[500:]))]


    assert len(train_lable_list) ==len(train_data_list)
    print len(train_lable_list)
    assert len(test_data_list) == len(test_lable_list)
    print len(test_data_list)

read_data()