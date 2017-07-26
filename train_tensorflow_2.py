# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 11:53:32 2017

@author: jaspe
"""

import tensorflow as tf
from tensorflow.estimator import Estimator, EstimatorSpec
import numpy as np

def cvt_y_to_onehot(y):
    return np.array([[0,1] if i == 1 else [0,1] for i in y]) 

def split_data(data, test_size=0.1):
            
    x = np.array([i[0] for i in data])
    y = np.array([i[1] for i in data])   
#    y = cvt_y_to_onehot(y)
    y = y.reshape(-1,1)    
    samples = len(x)
   
    x_train = x[:-int(samples*test_size)]
    x_test = x[-int(samples*test_size):]
    y_train = y[:-int(samples*test_size)]
    y_test = y[-int(samples*test_size):]
    print('data split')    
    return x_train, y_train, x_test, y_test

def load_and_split_data():
    data = np.load('training_data_balanced_tf.npy')
    print('data loaded')
    return split_data(data)

def model(features, labels, mode):
    n_classes = 2
    batch_size = 10000
    
    n_nodes_hl1 = 200
    n_nodes_hl2 = 50
    
    #placeholders for in and output data
    x = tf.placeholder('float', [None , 9600])
    y = tf.placeholder('float')
    
    # weights and biases of the hidden layers
    hidden_1_layer = {'weights': tf.Variable( tf.random_normal([9600, n_nodes_hl1])),
                      'biases': tf.Variable( tf.random_normal([n_nodes_hl1]) ) }
    
    hidden_2_layer = {'weights': tf.Variable( tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable( tf.random_normal([n_nodes_hl2]) ) }
    
    output_layer = {'weights': tf.Variable( tf.random_normal([n_nodes_hl2, n_classes])),
                    'biases': tf.Variable( tf.random_normal([n_classes]) ) }
    
    #propagation of data through the layers
    l1 = tf.add(tf.matmul(features['x'], hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']
    
    if (mode == tf.estimator.ModeKeys.TRAIN or
        mode == tf.estimator.ModeKeys.EVAL):
        loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y ) )
    else:
        loss = None
        
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = 
    else:
        train_op = None
    
    # Training sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    # ModelFnOps connects subgraphs we built to the
    # appropriate functionality.
    
    return tf.contrib.learn.ModelFnOps(mode=mode,
                                       predictions=y,
                                       loss=loss,
                                       train_op=train)




x_train, y_train, x_test, y_test = load_and_split_data()

feature_columns = [tf.contrib.layers.real_valued_column("x", dimension=9600)]

estimator = Estimator(model_fn= model,model_dir='./model/')

input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_test}, y_test)

# train
estimator.fit(input_fn=input_fn, steps=2000)
# Here we evaluate how well our model did. 
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r"% train_loss)
print("eval loss: %r"% eval_loss)