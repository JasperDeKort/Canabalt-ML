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

def play():
    
    #initialize variables
    start = time.time()
    dead = False
    consecutivedead = 0
    train_data = []
    savable = np.ones(4800)
    previous1 = np.zeros(4800)
    previous2 = np.zeros(4800)
    previous3 = np.zeros(4800)
    
    #press enter key to start the game
    print('starting to play')
    keypress.PressKey(0x1C)
    time.sleep(0.1)
    keypress.ReleaseKey(0x1C)
    
    while not dead:
        screen = grab_screen(region =(0,40,800,640))
#        score = screen[5:35,685:765]
#        score = cv2.threshold(score, 230,255,cv2.THRESH_BINARY)[1]
#        score1= score[:,20:40]
        savable, previous1, previous2, previous3 = process_image(screen), savable, previous1, previous2
        
        #savable = process_image(screen)        
        #X = savable.reshape(1,-1)       
        
        keys = key_check()
        output = keys_to_output(keys)
        train_data.append([savable, output])                               
        if np.array_equal(savable, previous3):
            if consecutivedead > 10:
                print('you are dead')
                dead = True
            consecutivedead += 1
        else:
            consecutivedead = 0        
#        cv2.imshow('processed view', savable)
#        if cv2.waitKey(25) & 0xFF == ord('q'):
#            cv2.destroyAllWindows()
#            break            
    run_time = time.time()-start        
    print('run took {} seconds, for {} screens, for {} seconds per screen'.format(run_time,
                                                                                  len(train_data),
                                                                                  (run_time/len(train_data))))
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
         
    for i in range(3,0,-1):
        print(i)
        time.sleep(1)    
    
    startdata = len(training_data)
    
    while True: 
        run_data, run_time = play()
        
        # disregard any short runs
        if run_time > 11:
            training_data += run_data  
        else:
            print('run too short, not saving')
        
        keys = key_check()
        if 'Q' in keys:
            print('q detected, hold if you want to quit')
            time.sleep(1)
            keys = key_check()
            if 'Q' in keys:
                print('saving before exit')
                np.save(file_name,training_data)
                print('saving done')
                break
        
        # only save data once every ... data points, 
        if len(training_data) > startdata + 20000:
            print('checkpoint reached, saving data')
            np.save(file_name,training_data)
            print('saved a total of {} samples'.format(len(training_data)))
            startdata = len(training_data)
        
        time.sleep(0.5)
      

         
if __name__ == "__main__":
    main()