# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 17:29:02 2017

@author: jaspe
"""

import numpy as np
import cv2
import time
from grabscreen import grab_screen
from getkeys import key_check
import os
from process_image import process_image
import pickle
import keypress

def keys_to_output(keys):
    if ' ' in keys:
        return 1
    else:
        return 0

def play(deadcheck):
    start = time.time()
    dead = False
    consecutivedead = 0
    train_data = []
    while not dead:
        screen = grab_screen(region =(0,40,800,640))
        savable = process_image(screen)        
        X = savable.reshape(1,-1)       
        
        keys = key_check()
        output = keys_to_output(keys)
#        print(output)
        train_data.append([savable, output])                        
        died = deadcheck.predict(X)        
        if died == 1:
            # only activate on 50 consecutive dead checks, to disregard false positives
            consecutivedead += 1
            if consecutivedead > 50:
                dead = True
        if died == 0:
            consecutivedead = 0
            
    run_time = time.time()-start       
        
    print('run took {} seconds, for {} screens, for {} seconds per screen'.format(run_time,
                                                                                  len(train_data),
                                                                                  (run_time/len(train_data))))
    # cut of first 200 screens as the start of the game is the same every time
    # cut of the last 250 screens as that playing resulted in death, and should not be learned by the network
    return train_data[:-250], run_time

def main():
    file_name = 'training_data.npy'
    
    if os.path.isfile(file_name):
        print('file exists. loading data')
        training_data = list(np.load(file_name))
    else:
        print('file does not exist. creating new file')
        training_data = []
    with open('MLPtrained_dead.pickle','rb') as f:
        deadcheck = pickle.load(f)
         
    for i in range(3,0,-1):
        print(i)
        time.sleep(1)    
    
    startdata = len(training_data)
    
    while True:
        print('starting to play')
        keypress.PressKey(0x1C)
        time.sleep(0.2)
        keypress.ReleaseKey(0x1C)
    
        run_data, run_time = play(deadcheck)
        
        # disregard any short runs
        if run_time > 11:
            training_data += run_data  
        else:
            print('run too short, not saving')
        
        # only save data once every ... data points, 
        if len(training_data) > startdata + 5000:
            np.save(file_name,training_data)
            print('saved a total of {} samples'.format(len(training_data)))
            startdata = len(training_data)
        
      

         
if __name__ == "__main__":
    main()