# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:53:34 2017

@author: jaspe
"""

import pickle
from sklearn.neural_network import MLPClassifier
from grabscreen import grab_screen
import numpy as np
import keypress
import cv2
from process_image import process_image
import time
import os
## following imports are for use in the self reinforcement loop. 
#from balance_data import balance_data
#from trainsklearn import train

file_name = 'training_data.npy'  

def play(mlp, deadcheck):
    start = time.time()
    dead = False
    consecutivedead = 0

## not saving the training data gives significant speed inmprovements. 
## replaced by a counter under the same name to preserve speed reports
#    train_data = []
    train_data = 0
    space = False
#    roundtime = time.time()
    while not dead:
               
        screen = grab_screen(region =(0,40,800,640))
        savable = process_image(screen)        
        X = savable.reshape(1,-1)       
        button = mlp.predict(X)
        
        if button == 1: 
            if space == False:
                keypress.PressKey(0x39)
                space = True
                print('space down')
        else:
            if space == True:
                keypress.ReleaseKey(0x39) 
                space = False
                print('space up')
            
#        train_data.append([savable, space])   
        train_data += 1                     
        died = deadcheck.predict(X)        
        if died == 1:
            consecutivedead += 1
            if consecutivedead > 50:
                dead = True
        if died == 0:
            consecutivedead = 0
        
#        timediff = time.time()-roundtime
#        if timediff < 0.023: 
#            time.sleep(0.01)
#            roundtime=time.time() 
        
    run_time = time.time()-start       
        
    print('run took {} seconds, for {} screens, for {} seconds per screen'.format(run_time,
                                                                                  train_data,
                                                                                  (run_time/train_data)))
#    return train_data[200:-100], run_time
    return None, run_time
    
              
def main():
    with open('iternumber.pickle','rb') as f:
        iternumber = pickle.load(f) 
        iternumber = 2
    
#    if os.path.isfile(file_name):
#        print('file exists. loading data')
#        training_data = list(np.load(file_name))
#    else:
#        print('file does not exist. creating new file')
#        training_data = []   
    print('opening: MLPtrained_{}.pickle'.format(iternumber - 1))
    with open('MLPtrained_{}.pickle'.format(iternumber - 1 ),'rb') as f:
        mlp = pickle.load(f)    
    with open('MLPtrained_dead.pickle','rb') as f:
        deadcheck = pickle.load(f)
        
    if os.path.isfile('run_times_{}.npy'.format(iternumber -1)):
        print('run times exists. loading data')
        run_times = list(np.load('run_times_{}.npy '.format(iternumber-1)))
    else:
        run_times = []
#        
#    startdata = len(training_data)
#    startdata2 = startdata

    for i in range(4,0,-1):
        print(i)
        time.sleep(1)
        
        
    while True:   
        print('starting to play')
        keypress.PressKey(0x1C)
        time.sleep(0.2)
        keypress.ReleaseKey(0x1C)
    
        run_data, run_time = play(mlp,deadcheck)
        run_times.append(run_time)
        np.save('run_times_{}.npy'.format(iternumber-1), run_times)
        
## the following code is for use in self reinforcement, currently commented out 
## as saving the data while playing results in a significantly increased loop time
## with this removed loop time is similar to data gathering loop time ( ~ 0.017 seconds per frame) 
#        
#        if run_time > 20:
#            training_data += run_data  
#        else:
#            print('run too short, not saving')
#        
#        if len(training_data) > (startdata2 + 5000):
#            np.save(file_name,training_data)
#            print('saved a total of {} samples'.format(len(training_data)))
#            startdata2 = len(training_data) 
#        
#        if len(training_data) > (startdata + 10000):
#            training_data = balance_data(training_data, overwrite=True)
#            mlp = train(mlp=mlp,train_data=training_data, iteration=str(iternumber))
#            np.save('run_times_{}.npy'.format(iternumber-1), run_times)
#            run_times = []
#            
#            startdata = len(training_data)
#            iternumber +=1
#            with open('iternumber.pickle','wb') as f:
#                pickle.dump(iternumber, f)
#    


     
        
if __name__ == "__main__":
    main()
        