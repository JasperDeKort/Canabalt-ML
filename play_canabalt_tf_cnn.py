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
import train_tensorflow_cnn

#import local files
from getkeys import key_check
from grabscreen import grab_screen
from process_image import process_image
from findwinrect import find_canabalt

tf.reset_default_graph()
## initialize variables from train_tensorflow_cnn so the model can be loaded.
n_classes = 2
batch_size = 100
log_folder = "./logs/cnngpu6_logs"
inputsize = [60, 80, 2]

# tweakable parameters
l2beta = 0.03
epsilon = 1
learning_rate = 0.01

testsize = 1000

input_keep = 0.8
layer_keep = 0.4
filtersize= 5
l1_outputchan = 16
l2_outputchan = 16
finallayer_in = 15*20*l2_outputchan
denselayernodes = 256

def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01, name=name), name=name)

# define filters and weights
filter1 = init_weights([filtersize,filtersize,inputsize[2], l1_outputchan], "filter_1")
filter2 = init_weights([filtersize,filtersize,l1_outputchan, l2_outputchan], "filter_2")
weights1 = init_weights([finallayer_in,denselayernodes],"weights_1")
weights2 = init_weights([denselayernodes,n_classes],"weights_2")

# dimensions of input and output
x = tf.placeholder('float', [None ,60,80,2], name='input_data')
y = tf.placeholder('float', [None, 2], name='output_data')
layer_keep_holder = tf.placeholder("float", name="input_keep")

def play(sess,prediction,screenloc,death):
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
        feed_dict={x:X, layer_keep_holder: 1}
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
        # counter to later calculate fps       
        train_data += 1
    # calculate run time and derivatives, and print them            
    run_time = time.time()-start       
    print('run took {} seconds, for {} screens, for {} seconds per screen'.format(run_time, train_data, (run_time/train_data)))
#    return train_data[200:-100], run_time
    return run_time
    
def main(): 
    # set up the tf graph for the model (no variables are loaded yet)     
    prediction = train_tensorflow_cnn.model(x, filter1,filter2 , layer_keep_holder, weights1, weights2)
    screenloc = find_canabalt()
    death = np.load("death250to280.npy")
    # count down before starting to give time to bring the game in focus
    for i in range(4,0,-1):
        print(i)
        time.sleep(1)
    with tf.Session() as sess:
        #reload trained tf model
        saver = tf.train.import_meta_graph(log_folder + '/canabalt_cnn-2361000.meta')
        saver.restore(sess, tf.train.latest_checkpoint(log_folder))
        #sess.run(tf.global_variables_initializer())
        while True: 
            #start to play
            run_time = play(sess,prediction,screenloc,death)
    
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
