#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import platform
import sys
BATCH_SIZE = 4
TEST_NUM = 2710

TEST_ITER = TEST_NUM/BATCH_SIZE
TEST_INTERVAL = 200

if platform.node() == 'm6':
    DATA_PATH = '/mnt/sdc/zihao.chen/data'
    sys.path.insert(0, '/usr/local/moji/caffe/python')
elif platform.node() == 'sentec-001':
    DATA_PATH = '/data/pig/'