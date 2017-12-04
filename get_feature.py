#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '/usr/local/moji/caffe/python')
import caffe
import os

caffe.set_mode_gpu()
model_def = 'net_file/deploy_inception-resnet-v2.prototxt'
model_weights = 'weights/inception-resnet-v2.caffemodel'

net = caffe.Net(model_def, model_weights, caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.array([128.0, 128.0, 128.0]))
# transformer.set_std('data', np.array([128.0, 128.0, 128.0]))
transformer.set_raw_scale('data', 255)
transformer.set_channel_swap('data', (2,1,0))

net.blobs['data'].reshape(4,3,299, 299)


image = caffe.io.load_image('data/train/image/9_995.png')
transformed_image = transformer.preprocess('data', image)

net.blobs['data'].data[...] = transformed_image
output = net.forward()
output_prob = net.blobs['pool_8x8_s1'].data[...]
print output_prob.shape
print net.blobs['pool_8x8_s1'].data[...][0,700,0,0]
print net.blobs['pool_8x8_s1'].data[...][1,700,0,0]
print net.blobs['pool_8x8_s1'].data[...][2,700,0,0]
print net.blobs['pool_8x8_s1'].data[...][3,700,0,0]