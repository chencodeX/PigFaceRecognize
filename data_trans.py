#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import lmdb
import caffe
import os
import cv2
import sys
import numpy as np
from glob import glob as gl
sys.path.insert("/usr/local/moji/caffe/python")
class_num = 30
train_image_path = '/mnt/sdc/zihao.chen/cloudRecognize/data/train/image'
def read_data():
    data_list=[]
    lable_list = []

    for class_index in range(class_num):
        all_pics_path = gl(os.path.join(train_image_path, '%d_*.png' % class_index))
        print len(all_pics_path)
