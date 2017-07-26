# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:02:29 2017

@author: jaspe
"""

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

n_classes = 2
batch_size = 10000

n_nodes_hl1 = 200
n_nodes_hl2 = 50
#n_nodes_hl3 = 50


# dimensions of input and output
x = tf.placeholder('float', [None , 9600])
y = tf.placeholder('float')

# hidden layers
hidden_1_layer = {'weights': tf.Variable( tf.random_normal([9600, n_nodes_hl1])),
                  'biases': tf.Variable( tf.random_normal([n_nodes_hl1]) ) }

hidden_2_layer = {'weights': tf.Variable( tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'biases': tf.Variable( tf.random_normal([n_nodes_hl2]) ) }

output_layer = {'weights': tf.Variable( tf.random_normal([n_nodes_hl2, n_classes])),
                'biases': tf.Variable( tf.random_normal([n_classes]) ) }

def neural_network_model(data):
    
    # propagation through the layers
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']
    
    return output

def train_neural_network(x_train, y_train, x_test, y_test):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y ) )

    # optimize, learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # number of cycles of feed forward and back propagation
    hm_epochs = 3
    saver = tf.train.Saver()
    
    print('starting training')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(x_train):
                start = i
                end = i+batch_size
                if end > len(x_train):
                    end = len(x_train)
                batch_x = np.array(x_train[start:end])
                batch_y = np.array(y_train[start:end])
                print('training samples {} to {}'.format(start, end))
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
            print('Epoch ', epoch + 1, ' completed out of ', hm_epochs, ' ,loss: ', epoch_loss)
            correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            print('Accuracy: ', accuracy.eval({x: x_test, y: y_test }) )
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Final accuracy: ', accuracy.eval({x: x_test, y: y_test }) )
        saver.save(sess, './canabalt_nn_25_5')

def split_data(data, test_size=0.1):
            
    x = np.array([i[0] for i in data])
    y = np.array([i[1] for i in data])   
    y = cvt_y_to_onehot(y)
    y = y.reshape(-1,2)    
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
  
def cvt_y_to_onehot(y):
    return np.array([[0,1] if i == 1 else [0,1] for i in y])      

def main():
    x_train, y_train, x_test, y_test = load_and_split_data()
    
    
    train_neural_network(x_train, y_train, x_test, y_test)
     

if __name__ == "__main__":
    main()