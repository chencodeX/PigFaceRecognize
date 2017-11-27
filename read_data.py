#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import pylab
import imageio
#注释的代码执行一次就好，以后都会默认下载完成
#imageio.plugins.ffmpeg.download()
import skimage
import numpy as np


class_num = 30

for video_index in range(class_num):
    #视频的绝对路径
    filename = 'data/train/video/%d.mp4'%(video_index+1)
    #可以选择解码工具
    vid = imageio.get_reader(filename,  'ffmpeg')
    for num,im in enumerate(vid):
        #image的类型是mageio.core.util.Image可用下面这一注释行转换为arrary
        save_url = 'data/train/image/%d_%d.png'%(video_index,num)
        print save_url
        imageio.imwrite(save_url,im)
        # image = skimage.img_as_float(im).astype(np.float64)
        # print image.shape
        # fig = pylab.figure()
        # fig.suptitle('image #{}'.format(num), fontsize=20)
        # pylab.imshow(im)
    # pylab.show()