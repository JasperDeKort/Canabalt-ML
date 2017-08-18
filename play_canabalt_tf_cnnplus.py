"""
Created on Wed Jun 28 11:53:34 2017

@author: jaspe
"""

import pickle
import numpy as np
import keypress
import cv2
import time
import os
import tensorflow as tf

#import local files
from getkeys import key_check
from grabscreen import grab_screen
from process_image import process_image
from findwinrect import find_canabalt

tf.reset_default_graph()
log_folder = "./logs/cnnplus_logs"
run_time_collect = []

def play(sess,screenloc,death):
    # initialize required variables and reset pressed buttons
    start = time.time()
    dead = False
    train_data = 0
    keypress.ReleaseKey(0x39)
    space = False   
    current = np.ones((60,80))
    previous1 = np.zeros((60,80))
    previous2 = np.zeros((60,80))
    previous3 = np.zeros((60,80))
    presult = [[0,0,0]]
    #restore the required placeholders for the feed dict
    graph = tf.get_default_graph()
    prediction = graph.get_tensor_by_name("hidden_4_dense/Relu:0")
    x = graph.get_tensor_by_name("input_data:0")
    x2 = graph.get_tensor_by_name("input_data2:0")
    layer_keep_holder =  graph.get_tensor_by_name("input_keep:0")
    # press enter to start the game
    print('starting to play')
    keypress.PressKey(0x1C)
    time.sleep(0.2)
    keypress.ReleaseKey(0x1C)
    time.sleep(0.2)
    while not dead:
        # grab screen and process             
        screen = grab_screen(region =screenloc)
        if np.array_equal(screen[:][250:280],death):
            dead = True
        current, previous1, previous2, previous3 = process_image(screen), current, previous1, previous2
        savable = np.stack([current,previous3],2)
        X = np.reshape(savable,(-1,60,80,2))
        # prepare image for prediction          
        feed_dict={x:X, x2:presult, layer_keep_holder: 1}
        result = prediction.eval(feed_dict, session=sess)
        # check prediction for chosen action
        if result[0][1] > result[0][0] :
            presult = [[1] + presult[0][0:2]]
            if space == False:
                keypress.PressKey(0x39)
                space = True
                
                #print('space down')
        else:
            presult = [[0] + presult[0][0:2]]
            if space == True:
                keypress.ReleaseKey(0x39) 
                space = False
                #print('space up')
        # counter to later calculate fps       
        train_data += 1
    # calculate run time and derivatives, and print them            
    run_time = time.time()-start       
    print('run took {} seconds, for {} screens, for {} seconds per screen'.format(run_time, train_data, (run_time/train_data)))
#    return train_data[200:-100], run_time
    return run_time
    
def main(): 
    # set up the tf graph for the model (no variables are loaded yet)     
    #prediction = train_tensorflow_cnn.model(x, filter1,filter2 , layer_keep_holder, weights1, weights2)
    screenloc = find_canabalt()
    death = np.load("death250to280.npy")
    
    # count down before starting to give time to bring the game in focus
    for i in range(4,0,-1):
        print(i)
        time.sleep(1)
    with tf.Session() as sess:
        #reload trained tf model
        saver = tf.train.import_meta_graph(log_folder + '/canabalt_cnn-2637000.meta')
        saver.restore(sess, tf.train.latest_checkpoint(log_folder))
        while True: 
            #start to play
            run_time = play(sess,screenloc,death)
            
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
