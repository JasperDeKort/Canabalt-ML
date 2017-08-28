# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 13:49:16 2017

@author: jaspe
"""

import tensorflow as tf
import numpy as np
import os

tf.reset_default_graph()


class Convnet():
    def __init__(self, name, logdir):
        self.name = name
        self.logdir = logdir
        self.batch_size = 100
        self.testsize = 1000
        self.n_classes = 2
        self.plusfeat = 3
        self.inputsize = [60, 80, 2]
        
        ## model layer properties
        self.layer_keep = 0.5
        self.filtersize = 5
        self.l1_outputchan = 16
        self.l2_outputchan = 24
        self.l3_outputchan = 32
        self.finallayer_in = (8*13*self.l3_outputchan)
        self.denselayernodes = 256
        self.f1shape = [self.filtersize, self.filtersize,
                        self.inputsize[2], self.l1_outputchan]
        self.f2shape = [self.filtersize, self.filtersize,
                        self.l1_outputchan, self.l2_outputchan]
        self.f3shape = [self.filtersize, self.filtersize,
                        self.l2_outputchan, self.l3_outputchan]
        self.w1shape = [self.finallayer_in + self.plusfeat,
                        self.denselayernodes]
        self.w2shape = [self.denselayernodes,self.n_classes]
        
        ## cost function settings
        self.epsilon = 0.3
        self.learning_rate = 0.003
        self.beta1 = 0.9
        self.beta2 = 0.95
        
        ## placeholders for input and output
        with tf.name_scope('input_data'):
            self.x = tf.placeholder('float', [None] + self.inputsize, name='x')
            self.x2 = tf.placeholder('float', [None, self.plusfeat], name='x2')
        with tf.name_scope('output_data'):
            self.y = tf.placeholder('float', [None, self.n_classes], name='y')
        self.layer_keep_holder = tf.placeholder("float", name="input_keep")
        
        ## filters and weights
        self.filter1 = init_weights(self.f1shape, "filter_1")
        self.filter2 = init_weights(self.f2shape, "filter_2")
        self.filter3 = init_weights(self.f3shape, "filter_3")
#        self.weights1 = init_weights(self.w1shape,"weights_1")
#        self.weights2 = init_weights(self.w2shape,"weights_2")
        
#        self.filter1 = tf.get_variable("filter_1", self.f1shape, 
#                                       initializer=tf.contrib.layers.xavier_initializer_conv2d())
#        self.filter2 = tf.get_variable("filter_2", self.f2shape, 
#                                       initializer=tf.contrib.layers.xavier_initializer_conv2d())
        self.weights1  = tf.get_variable("weights1", self.w1shape, initializer=tf.contrib.layers.xavier_initializer())
        self.weights2  = tf.get_variable("weights2", self.w2shape, initializer=tf.contrib.layers.xavier_initializer())
        
    
        ## model layers
        with tf.name_scope("hidden_1_conv"):
            self.conv1 = tf.nn.conv2d(self.x, self.filter1,
                                      strides=[1,1,1,1],padding="VALID")
            self.conv1rel = tf.nn.relu(self.conv1)
        with tf.name_scope("max_pooling_layer_1"):
            self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1rel, 
                                                 pool_size=[2 , 2], strides=2)
        with tf.name_scope("hidden_2_conv"):
            self.conv2 = tf.nn.conv2d(self.pool1, filter=self.filter2, 
                                      strides=[1,1,1,1],padding="VALID")
            self.conv2rel = tf.nn.relu(self.conv2)
        with tf.name_scope("max_pooling_layer_2"):
            self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2rel, 
                                                 pool_size=[2, 2], strides=2)
        with tf.name_scope("hidden_3_conv"):
            self.conv3 = tf.nn.conv2d(self.pool2, filter=self.filter3, 
                                      strides=[1,1,1,1],padding="VALID")
            self.conv3rel = tf.nn.relu(self.conv3)
        with tf.name_scope("hidden_4_dense"):
            self.pool2_flat = tf.reshape(self.conv3rel, [-1, self.finallayer_in])
            self.mergeflat = tf.concat([self.pool2_flat, self.x2],1)
#            self.dropout = tf.nn.dropout(self.mergeflat, self.layer_keep_holder)
            self.layer4 = tf.nn.relu(tf.matmul(self.mergeflat, self.weights1))
        with tf.name_scope("hidden_5_dense"):
            self.dropout2 = tf.nn.dropout(self.layer4, self.layer_keep_holder)  
            self.output = tf.nn.relu(tf.matmul(self.dropout2, self.weights2))
            
        ## cost function
        with tf.name_scope("cost"):
            self.cost = tf.losses.softmax_cross_entropy(onehot_labels=self.y, 
                                                        logits=self.output)
            self.optimize = tf.train.AdamOptimizer(self.learning_rate,
                                                    self.beta1,
                                                    self.beta2,
                                                    self.epsilon).minimize(self.cost)
#            self.train_op = self.optimizer..minimize(self.cost)
            
        ## accuracy calculation
        with tf.name_scope("acc_calc"):
            self.test_y = tf.argmax(self.y, 1)
            self.pred_y = tf.argmax(self.output,1)
            self.correct_pred = tf.equal(self.test_y, self.pred_y)
            self.acc_op = tf.reduce_mean(tf.cast(self.correct_pred, "float"))
        
        ## summary generation
        with tf.name_scope("test"):
        # generate summary of multiple accuracy metrics    
            tf.summary.scalar("accuracy", self.acc_op)
            self.conf_mat = tf.confusion_matrix(self.test_y, self.pred_y)
            tf.summary.scalar("true_negative", self.conf_mat[0][0])
            tf.summary.scalar("false_negative", self.conf_mat[1][0])
            tf.summary.scalar("true_positive", self.conf_mat[1][1])
            tf.summary.scalar("false_positive", self.conf_mat[0][1])
            tf.summary.scalar("cost", self.cost)
        
        ## generate summaries in train name scope to collect and merge easily        
        with tf.name_scope("train"):
            tf.summary.scalar("cost", self.cost)
            tf.summary.scalar("accuracy", self.acc_op)
                
        with tf.name_scope("weights"):
            tf.summary.histogram("weights_1", self.weights1)
            tf.summary.histogram("weights_2", self.weights2)
        
        ## generate images of the filters for human viewing
        with tf.variable_scope('visualization_filter1'):
            # to tf.image_summary format [batch_size, height, width, channels]
            self.kernel_transposed = tf.transpose (self.filter1, [3, 0, 1, 2])
            # reshape from 2 channel filters to 1 channel filters for image gen
            self.kernel_flattened = tf.reshape(self.kernel_transposed,
                                               [-1,self.filtersize,
                                                self.filtersize,1])
            tf.summary.image('conv1/filters', self.kernel_flattened, 
                             max_outputs=self.l1_outputchan*self.inputsize[2])
        
        ## generate images from filter pass through results. 1 for each class.        
        with tf.name_scope("f1_pass_through"):
            imageconvlist = []
            for i in range(self.n_classes):
                reshape = tf.reshape(self.conv1rel[i],
                                     [1,56,76,self.l1_outputchan])
                convtrans = tf.transpose( reshape, [3,1,2,0])
                imageconvlist.append(convtrans)        
                tf.summary.image('number_{}'.format(i), 
                                 imageconvlist[i],
                                 max_outputs = self.l1_outputchan)
        
        with tf.name_scope("f2_pass_through"):
            imageconvlist2 = []
            for i in range(self.n_classes):
                reshape = tf.reshape(self.conv2rel[i],
                                     [1,24,34,self.l2_outputchan])
                convtrans = tf.transpose(reshape, [3,1,2,0])
                imageconvlist2.append(convtrans)
                tf.summary.image('number_{}'.format(i), 
                                 imageconvlist2[i],
                                 max_outputs = self.l2_outputchan)
        
        with tf.name_scope("f3_pass_through"):
            imageconvlist3 = []
            for i in range(self.n_classes):
                reshape = tf.reshape(self.conv3rel[i],
                                     [1,8,13,self.l3_outputchan])
                convtrans = tf.transpose(reshape, [3,1,2,0])
                imageconvlist3.append(convtrans)
                tf.summary.image('number_{}'.format(i), 
                                 imageconvlist3[i],
                                 max_outputs = self.l3_outputchan)
                
        self.trainmerge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='train') + \
                                      tf.get_collection(tf.GraphKeys.SUMMARIES, scope='visualization_filter1') + \
                                      tf.get_collection(tf.GraphKeys.SUMMARIES, scope='weights'))
        self.testmerge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='test'))
        self.imagepassmerge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='f1_pass_through') + \
                                      tf.get_collection(tf.GraphKeys.SUMMARIES, scope='f2_pass_through') + \
                                      tf.get_collection(tf.GraphKeys.SUMMARIES, scope='f3_pass_through'))
        
        ## add desired tensors to collection for easy later restoring       
        tf.add_to_collection("prediction", self.output)
        tf.add_to_collection("optimizer", self.optimize)
        tf.add_to_collection("acc_op", self.acc_op )
        tf.add_to_collection("conf_mat", self.conf_mat )
        tf.add_to_collection("cost", self.cost )
        tf.add_to_collection("x", self.x )
        tf.add_to_collection("x2", self.x2 )
        tf.add_to_collection("y", self.y )
        
        ## record keeping
        self.saver = tf.train.Saver()
        self.i = 0

    def initialize(self, sess):
        if os.path.isfile( self.logdir + '/checkpoint'):
            print('previous version found. continguing')
            ckpt = tf.train.latest_checkpoint(self.logdir)
            self.saver.restore(sess,ckpt)
            #read checkpoint file and cast number at the end to int
            self.i = int(ckpt.split('-')[-1])
            self.writer = tf.summary.FileWriter(self.logdir, sess.graph)
        else:
            print('no previous model found. starting with untrained model')
            sess.run(tf.global_variables_initializer())
            self.i = 0
            self.writer = tf.summary.FileWriter(self.logdir, sess.graph)
                
    def train_iter(self, sess, x_batch, x2_batch, y_batch):
        feed_dict = {self.x: x_batch, 
                     self.x2: x2_batch, 
                     self.y: y_batch,
                     self.layer_keep_holder: self.layer_keep}
        _, cost = sess.run([self.optimize, self.cost], feed_dict)
        self.i += len(x_batch)
        return cost   
    
    def predict(self, sess, x, x2):
        feed_dict = {self.x: x, 
                     self.x2: x2,
                     self.layer_keep_holder: 1}
        prediction = sess.run(self.output, feed_dict = feed_dict)
        return prediction
    
    def accuracy(self, sess, x, x2, y):
        feed_dict = {self.x: x,
                     self.x2: x2, 
                     self.y: y,
                     self.layer_keep_holder: 1}
        accuracy = sess.run(self.acc_op, feed_dict = feed_dict)
        return accuracy
    
    def testsum(self, sess, x, x2, y):
        feed_dict = {self.x: x, 
                     self.x2: x2, 
                     self.y: y,
                     self.layer_keep_holder: 1}
        summary = sess.run(self.testmerge, feed_dict=feed_dict)
        self.writer.add_summary(summary,self.i)
        
    def trainsum(self, sess, x, x2, y):
        feed_dict = {self.x: x[:self.testsize], 
                     self.x2: x2[:self.testsize], 
                     self.y: y[:self.testsize],
                     self.layer_keep_holder: 1}
        summary = sess.run(self.trainmerge, feed_dict=feed_dict)
        self.writer.add_summary(summary,self.i)
    
    def imgsum(self, sess, x_img):
        feed_dict = {self.x: x_img}
        summary = sess.run(self.imagepassmerge, feed_dict=feed_dict)
        self.writer.add_summary(summary,self.i)        
        
    def save(self, sess):
        filename = self.logdir + '/' + self.name
        self.saver.save(sess, filename, global_step=self.i)
        self.writer.flush()
        
    def train(self, x_train, x2_train, y_train, 
              x_test, x2_test, y_test, x_img, epochs):
        print("starting training")
        with tf.Session() as sess:
            self.initialize(sess)
            for epoch in range(epochs):
                epoch_loss = 0
                previ = self.i
                for start, end in zip(range(0, len(x_train), self.batch_size), 
                                      range(self.batch_size, len(x_train)+1, 
                                            self.batch_size)):
                    batch_x = np.array(x_train[start:end])
                    batch_x2 = np.array(x2_train[start:end])
                    batch_y = np.array(y_train[start:end])
                    cost = self.train_iter(sess, batch_x, batch_x2, batch_y)
                    epoch_loss += cost
                    if self.i >= previ + 20000:
                        self.testsum(sess, x_test, x2_test, y_test)
                        self.trainsum(sess, x_train, x2_train, y_train)
                        self.imgsum(sess, x_img)
                        previ = self.i
                testaccuracy = self.accuracy(sess, x_test, x2_test, y_test)
                trainaccuracy = self.accuracy(sess, x_train[:self.testsize], 
                                              x2_train[:self.testsize], 
                                              y_train[:self.testsize])
                print("finished epoch {} of {}.".format(epoch+1, epochs) +
                      " test accuracy: {:.3%},".format(testaccuracy) + 
                      " train accuracy: {:.3%},".format(trainaccuracy) +
                      " epoch loss: {}".format(epoch_loss))
                self.save(sess)
            self.writer.close()
        return testaccuracy    


def init_weights(shape, name):
    # stddev gives best performance around 0.01. 
    # values of 0.4+ stop convergance
    with tf.name_scope(name):
#        tnorm = tf.truncated_normal(shape, stddev=0.001, name=name)
#        var = tf.Variable(tnorm, name=name)
#         var = tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(False))
         var = tf.get_variable(name, 
                               initializer=tf.truncated_normal(shape, 
                                                               stddev=0.001,name=name))
    return var

def split_data(data, testsize):            
    x = np.array([i[0] for i in data])
    x2 = np.array([i[1] for i in data])
    y = np.array([i[2] for i in data])   
    y = cvt_y_to_onehot(y)
    y = y.reshape(-1,2)    
    x_train = x[:-testsize]
    x2_train = x2[:-testsize]
    x_test = x[-testsize:]
    x2_test = x2[-testsize:]
    y_train = y[:-testsize]
    y_test = y[-testsize:]
    print('data split')  
    print("x_train shape: {}".format(x_train.shape))
    print("x2_train shape: {}".format(x2_train.shape))
    print("y_train shape: {}".format(y_train.shape))
    return x_train, x2_train, y_train, x_test, x2_test, y_test

def load_and_split_data(testsize):
    data = np.load('training_data_balanced_tf_cnn_plus.npy')
    print('data loaded')
    return split_data(data,testsize)
  
def cvt_y_to_onehot(y):
    return np.array([[1,0] if i==0 else [0,1] for i in y])  

def pick_image_for_class(x_test,y_test):
    x_images = []
    for i in range(len(y_test[1])):
        index = np.where(np.argmax(y_test,1)==i)[0][0]
        x_images.append( x_test[index])
    x_images = np.array(x_images)
    return x_images

#def build_and_train(x_train, x2_train, y_train, x_test, x2_test, y_test,x_img):
#    prediction, optimizer, acc_op, conf_mat, cost, x, x2, y, \
#    layer_keep_holder = create_graph(epsilon, beta1, beta2, learning_rate,
#                                      filtersize, l1_outputchan,l2_outputchan,
#                                      finallayer_in, denselayernodes,
#                                      n_classes,log_folder)
#        
#    accuracy = train_neural_network(x_train, x2_train, y_train, x_test, x2_test, y_test,x_img,
#                                    prediction, optimizer, acc_op, conf_mat,
#                                    cost, x, x2, y, layer_keep_holder,log_folder)
#    return accuracy    

def gridsearch(log_folder,x_train, x2_train, y_train, 
               x_test, x2_test, y_test,x_img):
    epsilons = [0.3, 1, 3]
    learning_rates = [0.003, 0.01]
    beta1s = [0.8, 0.9, 0.95]
    beta2s = [0.9, 0.95, 0.999]
    base_folder = "./logs/cnnplus_logs"
    results = {}
    
    for epsilon in epsilons:
        for learning_rate in learning_rates:
            for beta1 in beta1s:
                for beta2 in beta2s:
                    add_dir = "ep{}lr{}b1{}b2{}".format(epsilon,
                                                   learning_rate,beta1,beta2)
                    log_folder = base_folder + "/" + add_dir
                    print("training: " + add_dir)
                    if os.path.isfile(log_folder+ "/checkpoint"):
                        print("combo already done")
                    else:
                        accuracy = build_and_train(epsilon,beta1,beta2,
                                                   learning_rate,log_folder,
                                                   x_train, x2_train, y_train, 
                                                   x_test, x2_test,
                                                   y_test,x_img)                        
                        results[add_dir] = accuracy
    
    print(results)
    return results

def main():
    x_train, x2_train, y_train, x_test, x2_test, y_test = load_and_split_data(1000)   
    x_img = pick_image_for_class(x_test,y_test)    
    log_folder = "./logs/cnndropoutfix05x1pooling"
    model = Convnet('oopmod1',log_folder)    
    model.train(x_train, x2_train, y_train, x_test, x2_test, y_test,x_img, 15)     

if __name__ == "__main__":
    main()