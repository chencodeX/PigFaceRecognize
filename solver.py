#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import caffe
import surgery
from caffe.proto import caffe_pb2
import numpy as np
import os
from config import *


def denoise_solver(file_names, _lr, moment, _wd, fresh):
    s = caffe_pb2.SolverParameter()
    train_net_path = file_names[0]
    test_net_path = file_names[1]
    s.train_net = train_net_path
    s.test_net.append(test_net_path)
    s.test_iter.append(TEST_ITER)
    s.test_interval = TEST_INTERVAL
    if fresh:
        s.base_lr = _lr
    else:
        s.base_lr = _lr
    # s.lr_policy = "poly"
    s.lr_policy = "fixed"
    # s.power = 0.9
    # s.momentum = 0.95
    s.momentum = moment
    s.weight_decay = _wd
    # s.weight_decay = 0.0005
    s.display = 10
    s.average_loss = 10
    s.max_iter = 100000
    s.snapshot = 200
    s.type = "Nesterov"
    if fresh:
        _snap_path = os.path.join(SNAPSHOT_PATH, SUFFIX + NET_TYPE)
    else:
        _snap_path = os.path.join(SNAPSHOT_PATH, 'funtuning_' + SUFFIX + NET_TYPE)
    print 'snap_path:', _snap_path
    s.snapshot_prefix = _snap_path
    check_dir(snap_path)
    s.solver_mode = caffe_pb2.SolverParameter.GPU
    return s