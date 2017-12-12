#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from config import *

import cv2
import caffe
import os
import pickle
# from config import *
mean_value = np.array([128.0, 128.0, 128.0])
std = np.array([128.0, 128.0, 128.0])

crop_size = 299
base_size = 320
imag_root_path = os.path.join(DATA_PATH,'data/train/image/')
def image_preprocess(img):
    b, g, r = cv2.split(img)
    return cv2.merge([(b-mean_value[0])/std[0], (g-mean_value[1])/std[1], (r-mean_value[2])/std[2]])


def center_crop(img): # single crop
    short_edge = min(img.shape[:2])
    if short_edge < crop_size:
        return
    yy = int((img.shape[0] - crop_size) / 2)
    xx = int((img.shape[1] - crop_size) / 2)
    return img[yy: yy + crop_size, xx: xx + crop_size]

batch_size = 32
caffe.set_mode_gpu()
# caffe.set_device((0,1,2,3))
model_def = 'net_file/deploy_inception-resnet-v2-deploy.prototxt'
model_weights = 'weights/inception-resnet-v2.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)
net.blobs['data'].reshape(batch_size,3,299, 299)
# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_transpose('data', (2,0,1))
# transformer.set_mean('data', np.array([128.0, 128.0, 128.0]))
# # transformer.set_std('data', np.array([128.0, 128.0, 128.0]))
# transformer.set_raw_scale('data', 255)
# transformer.set_channel_swap('data', (2,1,0))
# image = caffe.io.load_image('data/train/image/9_995.png')


all_file_list =os.listdir(imag_root_path)[:1000]
print imag_root_path
print len(all_file_list)
all_file_num = len(all_file_list)
all_features = np.zeros((len(all_file_list),1536)).astype(np.float)

batch_num = all_file_num /128
for batch_index in range(batch_num+1):
    all_images=[]
    start, end = batch_index * batch_size, (batch_index + 1) * batch_size
    temp_all_file_list = all_file_list[start:end]
    for i in range(len(temp_all_file_list)):
        _img = cv2.imread(os.path.join(imag_root_path, temp_all_file_list[i]))
        _img = cv2.resize(_img, (int(_img.shape[1] * base_size / min(_img.shape[:2])),
                                 int(_img.shape[0] * base_size / min(_img.shape[:2]))))
        _img = image_preprocess(_img)
        _img = center_crop(_img)
        print _img.shape
        all_images.append(_img)

    all_images = np.array(all_images)
    print all_images.shape
    all_images = all_images.transpose(0, 3, 1, 2)
    net.blobs['data'].data[...] = all_images
    output = net.forward()
    output_prob = net.blobs['pool_8x8_s1'].data[...][:]
    print output_prob.shape
    # x =
    all_features[start:end,:]=output_prob[:,:,0,0]
    # print output_prob[0,:,0,0].shape
    # all_features.append(output_prob[0,:,0,0])
    # x =None

# print all_features

feature_map = {all_file_list[i]:all_features[i] for i in range(len(all_file_list))}
print feature_map
print len(feature_map)
f_f = open('inception_resnet_v2_feature_test.pkl','wb')
pickle.dump(feature_map,f_f)
f_f.close()


# net.blobs['data'].data[...] = _img
# output = net.forward()
# output_prob = net.blobs['pool_8x8_s1'].data[...]
# print output_prob.shape
# print net.blobs['pool_8x8_s1'].data[...][0,700,0,0]

# print net.blobs['pool_8x8_s1'].data[...][1,700,0,0]
# print net.blobs['pool_8x8_s1'].data[...][2,700,0,0]
# print net.blobs['pool_8x8_s1'].data[...][3,700,0,0]