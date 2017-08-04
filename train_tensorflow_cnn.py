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


inputsize = [60, 80, 1]

input_keep = 0.8
layer_keep = 0.5
l1_outputchan = 20

def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)

def model(input_data, filter1, layer_keep, weights1):
    # Add layer name scopes for better graph visualization
    with tf.name_scope("hidden_1_conv"):
        conv1 = tf.nn.conv2d(input_data, filter=filter1,strides=[1,1,1,1],padding="SAME",)
#    with tf.name_scope("max_pooling_layer_1"):
#        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2 , 2], strides=2)
#    with tf.name_scope("hidden_2_conv"):
#        conv2 = tf.layers.conv2d(inputs=pool1,
#                                 filters=20,
#                                 kernel_size=[5, 5],
#                                 padding="same",
#                                 activation=tf.nn.relu)
#    with tf.name_scope("max_pooling_layer_2"):
#        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    with tf.name_scope("hidden_3_dense"):
        pool2_flat = tf.reshape(conv1, [-1, 96000])
        dropout = tf.nn.dropout(pool2_flat, layer_keep)
        output = tf.nn.relu(tf.matmul(dropout, weights1))
        return output

# define filters and weights
filter1 = init_weights([7,7,inputsize[2], l1_outputchan], "filter_1")
weights1 = init_weights([96000,2],"weights_1")


# dimensions of input and output
x = tf.placeholder('float', [None ,60,80,1], name='input_data')
y = tf.placeholder('float', [None, 2], name='output_data')
layer_keep_holder = tf.placeholder("float", name="input_keep")

prediction = model(x, filter1, layer_keep_holder, weights1)

with tf.name_scope("cost"):
    # optimize, learning_rate = 0.001
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    tf.summary.scalar("cost", cost)
    
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1)) # Count correct predictions
    acc_op = tf.reduce_mean(tf.cast(correct_pred, "float")) # Cast boolean to float to average
    # Add scalar summary for accuracy
    tf.summary.scalar("accuracy", acc_op)

with tf.variable_scope('visualization'):
    # scale weights to [0 1], type is still float
    x_min = tf.reduce_min(filter1)
    x_max = tf.reduce_max(filter1)
    kernel_0_to_1 = (filter1 - x_min) / (x_max - x_min)

    # to tf.image_summary format [batch_size, height, width, channels]
    kernel_transposed = tf.transpose (kernel_0_to_1, [3, 0, 1, 2])

    # this will display random 3 filters from the 64 in conv1
    tf.summary.image('conv1/filters', kernel_transposed, max_outputs=20)

with tf.variable_scope('output_values'):
    class_max = tf.reduce_max(prediction, reduction_indices=[0])
    class_min = tf.reduce_min(prediction, reduction_indices=[0])
    class_mean = tf.reduce_mean(prediction, reduction_indices=[0])
    tf.summary.scalar('class_0_max', class_max[0])
    tf.summary.scalar('class_1_max', class_max[1])
    tf.summary.scalar('class_0_min', class_min[0])
    tf.summary.scalar('class_1_min', class_min[1])
    tf.summary.scalar('class_0_mean', class_mean[0])
    tf.summary.scalar('class_1_mean', class_mean[1])
    
def train_neural_network(x_train, y_train, x_test, y_test):
    print('test set size: {}'.format(len(y_test)))
    print('train set size: {}'.format(len(y_train)))
    # number of cycles of feed forward and back propagation
    hm_epochs = 1
    saver = tf.train.Saver()
    
    print('starting training')
    with tf.Session() as sess:   
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("./logs/cnn_logs", sess.graph) # for 1.0
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
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y,
                                                                layer_keep_holder: layer_keep})
                epoch_loss += c
                i += batch_size            
                summary, acc = sess.run([merged, acc_op], feed_dict={x: x_test, y: y_test,
                                                                    layer_keep_holder: 1})
                writer.add_summary(summary, i)
                if start % 1000 == 0:
                    print('current accuracy: {} at step {}'.format(acc, end))
            summary, acc = sess.run([merged, acc_op], feed_dict={x: x_test, y: y_test, layer_keep_holder: 1})
            writer.add_summary(summary, i)
            print('Epoch ', epoch + 1, ' completed out of ', hm_epochs, ' ,loss: ', epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Final accuracy: ', accuracy.eval({x: x_test, 
                                                 y: y_test,
                                                 layer_keep_holder: 1}) )
        saver.save(sess, './logs/cnn_logs/canabalt_cnn')
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
    data = np.load('training_data_balanced_tf_cnn.npy')
    print('data loaded')
    return split_data(data)
  
def cvt_y_to_onehot(y):
    return np.array([[1,0] if i==0 else [0,1] for i in y])      

def main():
    x_train, y_train, x_test, y_test = load_and_split_data()
    
    
    train_neural_network(x_train, y_train, x_test, y_test)
     

if __name__ == "__main__":
    main()