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
from findwinrect import find_canabalt



def keys_to_output(keys):
    if ' ' in keys:
        return 1
    else:
        return 0

def play(screenloc,death):
    
    #initialize variables
    start = time.time()
    dead = False
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
    time.sleep(0.2)
    
    while not dead:
        screen = grab_screen(region = screenloc)
        if np.array_equal(screen[:][250:280],death):
            dead = True
        savable, previous1, previous2, previous3 = process_image(screen), savable, previous1, previous2
        
        #savable = process_image(screen)        
        #X = savable.reshape(1,-1)       
        
        keys = key_check()
        output = keys_to_output(keys)
        train_data.append([savable, output])                               
      
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
    #file_name = 'training_data.npy'
    logdir = "./training_data/"
    runtype = "orig"
    if os.path.isfile(logdir + "runnumber.txt"):
        with open(logdir+ "runnumber.txt", 'r') as f:
            runnumber = int(f.read())
        runinfo = np.load(logdir + "runinfo.npy")
    else:
        runinfo = []
        runnumber = 1
        
    screenloc = find_canabalt()
    death = np.load("death250to280.npy")
         
    for i in range(3,0,-1):
        print(i)
        time.sleep(1)    
     
    while True: 
        run_data, run_time = play(screenloc,death)
        
#        # disregard any short runs
        if run_time > 11:
            runinfo = np.append(runinfo, [[runnumber, run_time, len(run_data), runtype]],0)
            np.save(logdir + "runinfo.npy", runinfo)            
            np.save(logdir + "rundata{}.npy".format(runnumber), run_data)
            runnumber+= 1
            with open(logdir + "runnumber.txt" , "w") as f:
                    f.write(str(runnumber)) 
        else:
            print('run too short, not saving')
        
        keys = key_check()
        if 'Q' in keys:
            print('q detected, hold if you want to quit')
            time.sleep(1)
            keys = key_check()
            if 'Q' in keys:
                with open(logdir + "runnumber.txt" , "w") as f:
                    f.write(str(runnumber))
                print('shutting down')
                break
        #sleep for half a second before restarting the game
        time.sleep(0.5)
      

         
if __name__ == "__main__":
    main()