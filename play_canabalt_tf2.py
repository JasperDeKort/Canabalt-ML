# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:53:34 2017

@author: jaspe
"""

import pickle
from grabscreen import grab_screen
import numpy as np
import keypress
import cv2
from process_image import process_image
import time
import os
import tensorflow as tf
import train_tensorflow
from tensorflow.contrib.learn import DNNClassifier, SKCompat
## following imports are for use in the self reinforcement loop. 
#from balance_data import balance_data
#from trainsklearn import train

file_name = 'training_data.npy'

def play(estimator, deadcheck):  
    start = time.time()
    dead = False
    consecutivedead = 0
    train_data = 0
    space = False   
    current = np.zeros(4800)
    previous1 = np.zeros(4800)
    previous2 = np.zeros(4800)
    previous3 = np.zeros(4800)
    
    
    print('starting to play')
    keypress.PressKey(0x1C)
    time.sleep(0.2)
    keypress.ReleaseKey(0x1C)
    while not dead:
               
        screen = grab_screen(region =(0,40,800,640))
        current, previous1, previous2, previous3 = process_image(screen), current, previous1, previous2
        savable = np.append(current,previous3)
        X = savable.reshape(1,-1)
        
        feed_dict={"x": X}
        result = estimator.predict(feed_dict)
        
        if train_data == 0:
            print(result)
        if result == 1 : 
            if space == False:
                keypress.PressKey(0x39)
                space = True
                #print('space down')
        else:
            if space == True:
                keypress.ReleaseKey(0x39) 
                space = False
                #print('space up')
               
        train_data += 1   
        if (train_data % 1000) == 0:
            print((time.time()-start)/1000)
            start = time.time()                   
        
    run_time = time.time()-start       
        
    print('run took {} seconds, for {} screens, for {} seconds per screen'.format(run_time,
                                                                                  train_data,
                                                                                  (run_time/train_data)))
#    return train_data[200:-100], run_time
    return None, run_time
    
              
def main():
    iternumber = 2
    with open('MLPtrained_dead.pickle','rb') as f:
        deadcheck = pickle.load(f)
        
    if os.path.isfile('run_times_{}.npy'.format(iternumber -1)):
        print('run times exists. loading data')
        run_times = list(np.load('run_times_{}.npy '.format(iternumber-1)))
    else:
        run_times = []
        
    feature_columns = [tf.contrib.layers.real_valued_column("x", dimension=9600)]
    estimator = SKCompat(DNNClassifier(feature_columns=feature_columns,
                              hidden_units=[256,64],
                              model_dir='./model/'))

    for i in range(4,0,-1):
        print(i)
        time.sleep(1)
                    
#    saver = tf.train.import_meta_graph('canabalt nn 200 50.meta')
    while True:    
        run_data, run_time = play(estimator, deadcheck)
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
        