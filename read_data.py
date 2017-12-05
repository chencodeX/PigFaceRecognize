#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import pylab
import cv2
import imageio
#注释的代码执行一次就好，以后都会默认下载完成
#imageio.plugins.ffmpeg.download()
import skimage
import numpy as np
from config import *

class_num = 30

for video_index in range(class_num):
    #视频的绝对路径
    filename = DATA_PATH+'data/train/video/%d.mp4'%(video_index+1)
    #可以选择解码工具
    vid = imageio.get_reader(filename,  'ffmpeg')
    for num,im in enumerate(vid):
        # if num %5 ==0:
            #image的类型是mageio.core.util.Image可用下面这一注释行转换为arrary
            save_url = DATA_PATH+'data/train/image/%d_%d.png'%(video_index,num)
            print save_url
            # imageio.imwrite(save_url,im)
            # print type(np.array(im).max())
            # image = skimage.img_as_ubyte(im)
            # print image.shape
            # print image[130:135,135:140,:]
            # print image.mean()
            # print image.max()
            # print image.min()
            im = np.array(im)
            im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

            cv2.imwrite(save_url,im)