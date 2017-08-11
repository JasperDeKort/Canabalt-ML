# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:02:29 2017

@author: jaspe
"""

import tensorflow as tf
import numpy as np

tf.reset_default_graph()

n_classes = 2
batch_size = 1000


inputsize = 9600
n_nodes_hl1 = 100
n_nodes_hl2 = 50
n_nodes_hl3 = 20

input_keep = 0.8
layer_keep = 0.5


# function to randomly initialize the weights
def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01, name=name), name=name)

def model(X, w_h, w_h2,w_h3, w_o, b_h, b_h2,b_h3, b_o, p_keep_input, p_keep_hidden):
    # Add layer name scopes for better graph visualization
    with tf.name_scope("layer1"):
        X = tf.nn.dropout(X, p_keep_input)
        h = tf.nn.relu(tf.add(tf.matmul(X, w_h), b_h))
    with tf.name_scope("layer2"):
        h = tf.nn.dropout(h, p_keep_hidden)
        h2 = tf.nn.relu(tf.add(tf.matmul(h, w_h2), b_h2))
    with tf.name_scope("layer3"):
        h = tf.nn.dropout(h2, p_keep_hidden)
        h3 = tf.nn.relu(tf.add(tf.matmul(h, w_h3), b_h3))
    with tf.name_scope("layer4"):
        h4 = tf.nn.dropout(h3, p_keep_hidden)
        return tf.add(tf.matmul(h4, w_o), b_o)

# randomly initialize weights using tf.Variable
w_h = init_weights([inputsize, n_nodes_hl1], "w_h")
w_h2 = init_weights([n_nodes_hl1, n_nodes_hl2], "w_h2")
w_h3 = init_weights([n_nodes_hl2, n_nodes_hl3], "w_h3")
w_o = init_weights([n_nodes_hl3, n_classes], "w_o")
b_h = init_weights([n_nodes_hl1],'b_h')
b_h2 = init_weights([n_nodes_hl2],'b_h2')
b_h3 = init_weights([n_nodes_hl3],'b_h3')
b_o = init_weights([n_classes],'b_o')


# generate histogram summaries for all weights
tf.summary.histogram("w_h_summ", w_h)
tf.summary.histogram("w_h2_summ", w_h2)
tf.summary.histogram("w_h3_summ", w_h3)
tf.summary.histogram("w_o_summ", w_o)
tf.summary.histogram("b_h_summ", b_h)
tf.summary.histogram("b_h2_summ", b_h2)
tf.summary.histogram("b_h3_summ", b_h3)
tf.summary.histogram("b_o_summ", b_o)

# dimensions of input and output
x = tf.placeholder('float', [None , inputsize], name='input_data')
y = tf.placeholder('float', name='output_data')
p_layer_keep = tf.placeholder("float", name="input_keep")
p_input_keep = tf.placeholder("float", name="layer_keep")

prediction = model(x,w_h,w_h2, w_h3 ,w_o, b_h, b_h2, b_h3, b_o, p_input_keep, p_layer_keep)

with tf.name_scope("cost"):
    # optimize, learning_rate = 0.001
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    tf.summary.scalar("cost", cost)
    
with tf.name_scope("accuracy"):
    test_y = tf.argmax(y, 1)
    pred_y = tf.argmax(prediction,1)
    correct_pred = tf.equal(test_y, pred_y) # Count correct predictions
    acc_op = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average
    tf.summary.scalar("accuracy", acc_op)
    conf_mat = tf.confusion_matrix(test_y, pred_y)
    tf.summary.scalar("true_negative", conf_mat[0][0])
    tf.summary.scalar("false_negative", conf_mat[1][0])
    tf.summary.scalar("true_positive", conf_mat[1][1])
    tf.summary.scalar("false_positive", conf_mat[0][1])
   
def train_neural_network(x_train, y_train, x_test, y_test):
    print('test set size: {}'.format(len(x_test)))
    print('train set size: {}'.format(len(x_train)))
    # number of cycles of feed forward and back propagation
    hm_epochs = 20
    saver = tf.train.Saver()
    
    print('starting training')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./logs/nn3_logs", sess.graph) # for 1.0
        merged = tf.summary.merge_all()
        
        i = 0
        for epoch in range(hm_epochs):
            epoch_loss = 0
            
            for start, end in zip(range(0, len(x_train), batch_size), range(batch_size, len(x_train)+1, batch_size)):
                if end > len(x_train):
                    end = len(x_train)
                batch_x = np.array(x_train[start:end])
                batch_y = np.array(y_train[start:end])
                #print('training samples {} to {}'.format(start, end))
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, 
                                                                y: batch_y,
                                                                p_input_keep: input_keep, 
                                                                p_layer_keep: layer_keep})
                summary, acc = sess.run([merged, acc_op], feed_dict={x: x_test, y: y_test,
                                        p_input_keep: 1.0, p_layer_keep: 1.0})
                writer.add_summary(summary, i)
                epoch_loss += c
                i += batch_size
            
                summary, acc = sess.run([merged, acc_op], feed_dict={x: x_test, y: y_test,
                                        p_input_keep: 1.0, p_layer_keep: 1.0})
                writer.add_summary(summary, i)
                if start % 10000 == 0:
                    print('step: {} ,current accuracy: {}'.format(start,acc))
            print('Epoch ', epoch + 1, ' completed out of ', hm_epochs, ' ,loss: ', epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Final accuracy: ', accuracy.eval({x: x_test, 
                                                 y: y_test,
                                                 p_input_keep: 1, 
                                                 p_layer_keep: 1 }) )
        saver.save(sess, './logs/nn3_logs/canabalt_nn_100_50_20')
        writer.flush()
        writer.close()

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
    return np.array([[0,1] if i == 1 else [1,0] for i in y])      

def main():
    x_train, y_train, x_test, y_test = load_and_split_data()
    
    
    train_neural_network(x_train, y_train, x_test, y_test)
     

if __name__ == "__main__":
    main()