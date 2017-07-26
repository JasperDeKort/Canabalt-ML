# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 11:53:34 2017

@author: jaspe
"""

import pickle
from grabscreen import grab_screen
import numpy as np
import keypress
from getkeys import key_check
import cv2
from process_image import process_image
import time
import os
import tensorflow as tf
import train_tensorflow
## following imports are for use in the self reinforcement loop. 
#from balance_data import balance_data
#from trainsklearn import train

file_name = 'training_data.npy'  

def play(prediction):
    with tf.Session() as sess:
        #reload trained tf model
        saver = tf.train.import_meta_graph('./canabalt_nn_25_5.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))
        sess.run(tf.global_variables_initializer())
        # initialize required variables
        start = time.time()
        dead = False
        consecutivedead = 0
        train_data = 0
        space = False   
        current = np.ones(4800)
        previous1 = np.zeros(4800)
        previous2 = np.zeros(4800)
        previous3 = np.zeros(4800)
        # press enter to start the game
        print('starting to play')
        keypress.PressKey(0x1C)
        time.sleep(0.2)
        keypress.ReleaseKey(0x1C)
        while not dead:
            # grab screen and process             
            screen = grab_screen(region =(0,40,800,640))
            current, previous1, previous2, previous3 = process_image(screen), current, previous1, previous2
            savable = np.append(current,previous3)
            # prepare image for prediction 
            X = savable.reshape(1,-1)          
            feed_dict={train_tensorflow.x:X}
            result = prediction.eval(feed_dict, session=sess)
            # check prediction for chosen action
            if result[0][1] > result[0][0] : 
                if space == False:
                    keypress.PressKey(0x39)
                    space = True
                    #print('space down')
            else:
                if space == True:
                    keypress.ReleaseKey(0x39) 
                    space = False
                    #print('space up')
            # check if the player has died by detecting changes in the screen. no changes == dead
            if np.array_equal(current, previous3) or np.array_equal(current, previous1):
                if consecutivedead > 20:
                    print('i am dead, did i do well?')
                    dead = True
                consecutivedead += 1    
            else:
                consecutivedead = 0
            # counter to later calculate fps       
            train_data += 1
    # calculate run time and derivatives, and print them            
    run_time = time.time()-start       
    print('run took {} seconds, for {} screens, for {} seconds per screen'.format(run_time, train_data, (run_time/train_data)))
#    return train_data[200:-100], run_time
    return None, run_time
    
def main():
    # check which iteration of the model this is, and reload run times for recording
    iternumber = 2    
    if os.path.isfile('run_times_{}.npy'.format(iternumber -1)):
        print('run times exists. loading data')
        run_times = list(np.load('run_times_{}.npy '.format(iternumber-1)))
    else:
        run_times = []    
    # set up the tf graph for the model (no variables are loaded yet)     
    prediction = train_tensorflow.neural_network_model(train_tensorflow.x)
    # count down before starting to give time to bring the game in focus
    for i in range(4,0,-1):
        print(i)
        time.sleep(1)
    while True: 
        #start to play
        run_data, run_time = play(prediction)
        # append runtimes to previous data and save
        run_times.append(run_time)
        np.save('run_times_{}.npy'.format(iternumber-1), run_times)
        # check if q button is pressed, if pressed -> shut down
        keys = key_check()
        if 'Q' in keys:
            print('q detected, hold if you want to quit')
            time.sleep(1)
            keys = key_check()
            if 'Q' in keys:
                print('quitting')
                break
        
if __name__ == "__main__":
    main()
        