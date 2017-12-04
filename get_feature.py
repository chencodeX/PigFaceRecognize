#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/usr/local/moji/caffe/python')
import cv2
import caffe
import os

mean_value = np.array([128.0, 128.0, 128.0])
std = np.array([128.0, 128.0, 128.0])

crop_size = 299
base_size = 320
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


caffe.set_mode_gpu()
model_def = 'net_file/deploy_inception-resnet-v2.prototxt'
model_weights = 'weights/inception-resnet-v2.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)
# transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
# transformer.set_transpose('data', (2,0,1))
# transformer.set_mean('data', np.array([128.0, 128.0, 128.0]))
# # transformer.set_std('data', np.array([128.0, 128.0, 128.0]))
# transformer.set_raw_scale('data', 255)
# transformer.set_channel_swap('data', (2,1,0))

net.blobs['data'].reshape(1,3,299, 299)


# image = caffe.io.load_image('data/train/image/9_995.png')
_img = cv2.imread('data/train/image/9_990.png')
_img = cv2.resize(_img, (int(_img.shape[1] * base_size / min(_img.shape[:2])),
                                 int(_img.shape[0] * base_size / min(_img.shape[:2])))
                          )
_img = image_preprocess(_img)
_img = center_crop(_img)
_img = _img[np.newaxis,...]
# transformed_image = transformer.preprocess('data', image)
_img = _img.transpose(0, 3, 1, 2)
net.blobs['data'].data[...] = _img
output = net.forward()
output_prob = net.blobs['pool_8x8_s1'].data[...]
print output_prob.shape
print net.blobs['pool_8x8_s1'].data[...][0,700,0,0]
# print net.blobs['pool_8x8_s1'].data[...][1,700,0,0]
# print net.blobs['pool_8x8_s1'].data[...][2,700,0,0]
# print net.blobs['pool_8x8_s1'].data[...][3,700,0,0]