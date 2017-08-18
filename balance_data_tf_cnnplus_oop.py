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
        self.input_keep = 0.8
        self.layer_keep = 0.4
        self.filtersize= 5
        self.l1_outputchan = 16
        self.l2_outputchan = 16
        self.finallayer_in = (15*20*self.l2_outputchan)
        self.denselayernodes = 256
        
        ## cost function settings
        self.epsilon = 0.3
        self.learning_rate = 0.003
        self.beta1 = 0.9
        self.beta2 = 0.95
        
        ## placeholders for input and output
        self.x = tf.placeholder('float', [None] + self.inputsize, name='input_data')
        self.x2 = tf.placeholder('float', [None] + [self.plusfeat] , name='input_data2')
        self.y = tf.placeholder('float', [None] + [self.n_classes], name='output_data')
        
        ## filters and weights
        self.filter1 = init_weights([self.filtersize,self.filtersize,self.inputsize[2], self.l1_outputchan], "filter_1")
        self.filter2 = init_weights([self.filtersize,self.filtersize,self.l1_outputchan, self.l2_outputchan], "filter_2")
        self.weights1 = init_weights([self.finallayer_in + self.plusfeat,self.denselayernodes],"weights_1")
        self.weights2 = init_weights([self.denselayernodes,self.n_classes],"weights_2")
        
        ## model layers
        with tf.name_scope("hidden_1_conv"):
            self.conv1 = tf.nn.conv2d(self.x, self.filter1,strides=[1,1,1,1],padding="SAME")
        with tf.name_scope("max_pooling_layer_1"):
            self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2 , 2], strides=2)
        with tf.name_scope("hidden_2_conv"):
            self.conv2 = tf.nn.conv2d(self.pool1, filter=self.filter2,strides=[1,1,1,1],padding="SAME")
        with tf.name_scope("max_pooling_layer_2"):
            self.pool2 = tf.layers.max_pooling2d(inputs=self.conv2, pool_size=[2, 2], strides=2)
        with tf.name_scope("hidden_3_dense"):
            self.pool2_flat = tf.reshape(self.pool2, [-1, self.finallayer_in])
            self.mergeflat = tf.concat([self.pool2_flat, self.x2],1)
            self.layer3 = tf.nn.relu(tf.matmul(self.mergeflat, self.weights1))
            self.dropout = tf.nn.dropout(self.layer3, self.layer_keep)
        with tf.name_scope("hidden_4_dense"):    
            self.output = tf.nn.relu(tf.matmul(self.dropout, self.weights2))
        
        ## cost function
        with tf.name_scope("cost"):
            self.cost = tf.losses.softmax_cross_entropy(onehot_labels=self.y, logits=self.output)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    epsilon=self.epsilon, beta1=self.beta1,
                                                    beta2=self.beta2).minimize(self.cost)
        ## accuracy calculation
        self.test_y = tf.argmax(self.y, 1)
        self.pred_y = tf.argmax(self.output,1)
        self.correct_pred = tf.equal(self.test_y, self.pred_y) # Count correct predictions
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
        
            # generate summaries in train name scope to collect and merge easily        
        with tf.name_scope("train"):
            tf.summary.scalar("cost", self.cost)
            tf.summary.scalar("accuracy", self.acc_op)
                
        with tf.name_scope("weights"):
            tf.summary.histogram("weights_1", self.weights1)
            tf.summary.histogram("weights_2", self.weights2)
        
        # generate images of the filters for human viewing
        with tf.variable_scope('visualization_filter1'):
            # to tf.image_summary format [batch_size, height, width, channels]
            self.kernel_transposed = tf.transpose (self.filter1, [3, 0, 1, 2])
            # reshape from 2 channel filters to 1 channel filters for image gen
            self.kernel_flattened = tf.reshape(self.kernel_transposed,
                                               [-1,self.filtersize,
                                                self.filtersize,1])
            tf.summary.image('conv1/filters', self.kernel_flattened, 
                             max_outputs=self.l1_outputchan*self.inputsize[2])
        
        # generate images from filter pass through results. 1 for each class.        
        with tf.name_scope("f1pass"):
            self.imageconvlist = []
            self.imageconv1 = tf.nn.relu(tf.nn.conv2d(self.x, self.filter1,strides=[1,1,1,1],padding="SAME"))
            for i in range(self.n_classes):
                self.imageconvlist.append(tf.transpose(tf.reshape(self.imageconv1[i],[1,60,80,self.l1_outputchan]), [3,1,2,0]))        
                tf.summary.image('number_{}'.format(i), self.imageconvlist[i],max_outputs = self.l1_outputchan)
        
        with tf.name_scope("f2pass"):
            self.imageconvlist2 = []
            self.imagepool1 = tf.layers.max_pooling2d(inputs=self.imageconv1, pool_size=[2 , 2], strides=2)
            self.imageconv2 = tf.nn.relu(tf.nn.conv2d(self.imagepool1, self.filter2,strides=[1,1,1,1],padding="SAME"))
            for i in range(self.n_classes):
                self.imageconvlist2.append(tf.transpose(tf.reshape(self.imageconv2[i],[1,30,40,self.l2_outputchan]), [3,1,2,0]))
                tf.summary.image('number_{}'.format(i), self.imageconvlist2[i],max_outputs = self.l2_outputchan)
        
        ## add desired tensors to collection for easy later restoring       
        tf.add_to_collection("prediction", self.output)
        tf.add_to_collection("optimizer", self.optimizer)
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
            self.saver.restore(sess,tf.train.latest_checkpoint(self.logdir))
            #read checkpoint file and cast number at the end to int
            ckpt = tf.train.get_checkpoint_state(self.logdir)
            self.i = int(str(ckpt).split('-')[-1][:-2])
            self.writer = tf.summary.FileWriter(self.logdir, sess.graph)
        else:
            print('no previous model found. starting with untrained model')
            sess.run(tf.global_variables_initializer())
            self.i = 0
            self.writer = tf.summary.FileWriter(self.logdir, sess.graph)
                
    def train_iter(self, sess, x_batch, x2_batch, y_batch):
        feed_dict = {self.x: x_batch, self.x2: x2_batch, self.y: y_batch}
        _, cost = sess.run([self.optimizer, self.cost], feed_dict)
        self.i += len(x_batch)
        return cost   
    
    def predict(self, sess, x, x2):
        feed_dict = {self.x:x, self.x2:x2}
        prediction = sess.run(self.output, feed_dict = feed_dict)
        return prediction
    
    def accuracy(self, sess, x, x2, y):
        feed_dict = {self.x: x, self.x2: x2, self.y: y}
        accuracy = sess.run(self.acc_op, feed_dict = feed_dict)
        return accuracy
        
    def save(self, sess):
        self.saver.save(sess, self.logdir + '/' + self.name, global_step=self.i)
    
    


def init_weights(shape, name):
    # stddev gives best performance around 0.01. values of 0.4+ stop convergance
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01, name=name), name=name)

def train_neural_network(x_train, x2_train, y_train, x_test, x2_test, y_test,x_img, model):
    # number of cycles of feed forward and back propagation
    hm_epochs = 5
    batch_size = 100
    print('starting training')
    with tf.Session() as sess:   
        model.initialize(sess)
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for start, end in zip(range(0, len(x_train), batch_size), range(batch_size, len(x_train)+1, batch_size)):
                if end > len(x_train):
                    end = len(x_train)
                batch_x = np.array(x_train[start:end])
                batch_x2 = np.array(x2_train[start:end])
                batch_y = np.array(y_train[start:end])
                c = model.train_iter(sess, batch_x, batch_x2, batch_y)
                epoch_loss += c
            accuracy = model.accuracy(sess, x_test, x2_test, y_test)
            print("finished epoch {} of {}. current accuracy: {}".format(epoch+1, hm_epochs, accuracy))
            model.save(sess)
    return accuracy

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

def gridsearch(log_folder,x_train, x2_train, y_train, x_test, x2_test, y_test,x_img):
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
    log_folder = "./logs/cnnoop2_logs"
    model = Convnet('oopmod1',log_folder)    
    accuracy = train_neural_network(x_train, x2_train, y_train, x_test, x2_test, y_test,x_img, model)
    return accuracy
     

if __name__ == "__main__":
    main()