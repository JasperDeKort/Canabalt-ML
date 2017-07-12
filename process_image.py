# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:37:03 2017

@author: jaspe
"""

import numpy as np
import cv2


opener = np.ones((2,7),np.uint8)
dialator = np.ones((10,8),np.uint8)

def process_image(original_image):
    processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.threshold(processed_image, 240,255,cv2.THRESH_BINARY)[1]
#    processed_image = cv2.morphologyEx(processed_image, cv2.MORPH_OPEN, opener)
    processed_image = cv2.dilate(processed_image, dialator, iterations=1)
    processed_image = cv2.resize(processed_image,(80,60))
    processed_image = cv2.threshold(processed_image, 30,255,cv2.THRESH_BINARY)[1]
    processed_image = processed_image.reshape(4800)
#    processed_image = cv2.Canny(processed_image, threshold1=350, threshold2=450)
    return processed_image