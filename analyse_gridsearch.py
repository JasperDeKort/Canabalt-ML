# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 20:28:54 2017

@author: jaspe
"""

import os
import numpy as np
import tensorflow as tf
import train_tensorflow_cnn as ttc

input_keep = 0.8
layer_keep = 0.4
filtersize= 5
l1_outputchan = 16
l2_outputchan = 16
finallayer_in = 15*20*l2_outputchan
denselayernodes = 256
n_classes = 2

base_folder = "./logs/cnngpu_logs"
folders = os.listdir(base_folder)

results = []

x_train, y_train, x_test, y_test = ttc.load_and_split_data()
print("testing on {} values in x and y test".format(len(y_test)))

for folder in folders:
    tf.reset_default_graph()
    log_folder = base_folder + "/" + folder
    epsilon = 1
    beta1 = 0.9
    beta2 = 0.999
    learning_rate = 0.001

    if os.path.isfile(log_folder + '/canabalt_cnn-1263000.meta'):
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(log_folder + '/canabalt_cnn-1263000.meta')
            saver.restore(sess, tf.train.latest_checkpoint(log_folder))
            graph = tf.get_default_graph()
#            for op in tf.get_default_graph().get_operations():
#                if "confusion_matrix" in str(op.values()):
#                    print( op.values() )
            
            acc_op = graph.get_tensor_by_name("test/accuracy/Mean:0")
            cost =  graph.get_tensor_by_name("cost/softmax_cross_entropy_loss/value:0")
            conf_mat =  graph.get_tensor_by_name("test/accuracy/confusion_matrix/SparseTensorDenseAdd:0")
            x = graph.get_tensor_by_name("input_data:0")
            y =  graph.get_tensor_by_name("output_data:0")
            layer_keep_holder =  graph.get_tensor_by_name("input_keep:0")
            
            feed_dict = {x:x_test, y:y_test, layer_keep_holder: 1}
            accuracy, loss , confusion = sess.run([acc_op, cost, conf_mat],feed_dict)
            print("{}    {}     {}".format(folder, accuracy, loss))
            print(confusion)
            results += [[folder, accuracy, loss, confusion]]

results.sort(key=lambda x: x[1],reverse=True)
print(results[0:2][0:2])
            
        
