#!/usr/bin/evn python
# -*- coding: utf-8 -*-
import cv2
import numpy as np

# video_file = open('data/train/1.mp4','r')

cap = cv2.VideoCapture('data/train/1.mp4')
print cap
while(1):
    ret, frame = cap.read()
    print frame.shape
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('frame', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

cap.release()
# cv2.destroyAllWindows()