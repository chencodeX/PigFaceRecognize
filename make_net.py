#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop

def conv_relu(bottom,num_out,ks=3,stride=1,pad=1,c_double=4,dilation):
    pass