# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:29:02 2017

@author: jaspe
"""

import numpy as np
import cv2
import keypress
import time
from grabscreen import grab_screen
from getkeys import key_check
import os

def keys_to_output(keys):
    if ' ' in keys:
        return 1
    else:
        return 0
    

def process_image(original_image):
    processed_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    processed_image = cv2.threshold(processed_image, 240,255,cv2.THRESH_BINARY)[1]
#    processed_image = cv2.Canny(processed_image, threshold1=350, threshold2=450)
    return processed_image

file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('file exists. loading data')
    training_data = list(np.load(file_name))
else:
    print('file does not exist. creating new file')
    training_data = []

def main():
    for i in range(5,0,-1):
        print(i)
        time.sleep(1)
    
    last_time = time.time()
    while True:
        screen = grab_screen(region =(0,40,800,640))
        
        processed_image = process_image(screen)
        processed_image = cv2.resize(processed_image,(160,128))
        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([processed_image, output])
               
        if len(training_data) % 500 == 0:
            np.save(file_name,training_data)
            print('loop took {} seconds'.format(time.time() - last_time))
            last_time = time.time()
        cv2.imshow('processed view', processed_image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
         
if __name__ == "__main__":
    main()
