# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 16:02:29 2017

@author: jaspe
"""

import tensorflow as tf
import numpy as np
import os

tf.reset_default_graph()

batch_size = 100



inputsize = [60, 80, 2]

# tweakable parameters
#l2beta = 0.03


testsize = 1000

input_keep = 0.8
layer_keep = 0.4
filtersize= 5
l1_outputchan = 16
l2_outputchan = 16
finallayer_in = 15*20*l2_outputchan
denselayernodes = 256
n_classes = 2

def init_weights(shape, name):
    # stddev gives best performance around 0.01. values of 0.4+ stop convergance
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01, name=name), name=name)

def model(input_data, filter1,filter2 , layer_keep, weights1, weights2):
    # Add layer name scopes for better graph visualization
    with tf.name_scope("hidden_1_conv"):
        conv1 = tf.nn.conv2d(input_data, filter1,strides=[1,1,1,1],padding="SAME")
    with tf.name_scope("max_pooling_layer_1"):
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2 , 2], strides=2)
    with tf.name_scope("hidden_2_conv"):
        conv2 = tf.nn.conv2d(pool1, filter=filter2,strides=[1,1,1,1],padding="SAME")
    with tf.name_scope("max_pooling_layer_2"):
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    with tf.name_scope("hidden_3_dense"):
        pool2_flat = tf.reshape(pool2, [-1, finallayer_in])
        layer3 = tf.nn.relu(tf.matmul(pool2_flat, weights1))
        dropout = tf.nn.dropout(layer3, layer_keep)
    with tf.name_scope("hidden_4_dense"):    
        output = tf.nn.relu(tf.matmul(dropout, weights2))
        return output

def create_graph(epsilon, beta1, beta2, learning_rate,filtersize, l1_outputchan,
                 l2_outputchan, finallayer_in, denselayernodes,n_classes, log_folder):
    tf.reset_default_graph()
    # define filters and weights
    filter1 = init_weights([filtersize,filtersize,inputsize[2], l1_outputchan], "filter_1")
    filter2 = init_weights([filtersize,filtersize,l1_outputchan, l2_outputchan], "filter_2")
    weights1 = init_weights([finallayer_in,denselayernodes],"weights_1")
    weights2 = init_weights([denselayernodes,n_classes],"weights_2")
    
    # dimensions of input and output
    x = tf.placeholder('float', [None] + inputsize, name='input_data')
    y = tf.placeholder('float', [None] + [n_classes], name='output_data')
    layer_keep_holder = tf.placeholder("float", name="input_keep")
    
    prediction = model(x, filter1,filter2 , layer_keep_holder, weights1, weights2)
    
    with tf.name_scope("cost"):
        # optimize, learning_rate = 0.001
        #regularization = tf.nn.l2_loss(filter1)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=prediction)
        cost = loss #+ l2beta * regularization
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           epsilon=epsilon, beta1=beta1,
                                           beta2=beta2).minimize(cost)
        
    
    with tf.name_scope("test"):
        # generate summary of multiple accuracy metrics    
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
            tf.summary.scalar("cost", cost)
            
        # generate over time summary of min, max and mean predicted values for each class
        with tf.variable_scope('output_values'):
            class_max = tf.reduce_max(prediction, reduction_indices=[0])
            class_mean = tf.reduce_mean(prediction, reduction_indices=[0])
            tf.summary.scalar('class_0_max', class_max[0])
            tf.summary.scalar('class_1_max', class_max[1])
            tf.summary.scalar('class_0_mean', class_mean[0])
            tf.summary.scalar('class_1_mean', class_mean[1])
    
    # generate summaries in train name scope to collect and merge easily        
    with tf.name_scope("train"):
        with tf.name_scope("cost"):
            tf.summary.scalar("cost", cost)
        
        # generate summary of multiple accuracy metrics    
        with tf.name_scope("accuracy"):
            tf.summary.scalar("accuracy", acc_op)
            
    with tf.name_scope("weights"):
        tf.summary.histogram("weights_1", weights1)
        tf.summary.histogram("weights_2", weights2)
    
    # generate images of the filters for human viewing
    with tf.variable_scope('visualization_filter1'):
        # to tf.image_summary format [batch_size, height, width, channels]
        kernel_transposed = tf.transpose (filter1, [3, 0, 1, 2])
        # reshape from 2 channel filters to 1 channel filters for image gen
        kernel_flattened = tf.reshape(kernel_transposed,[-1,filtersize,filtersize,1])
        tf.summary.image('conv1/filters', kernel_flattened, max_outputs=l1_outputchan*inputsize[2])
    
    # generate images from filter pass through results. 1 for each class.        
    with tf.name_scope("f1pass"):
        imageconvlist = []
        imageconv1 = tf.nn.relu(tf.nn.conv2d(x, filter1,strides=[1,1,1,1],padding="SAME"))
        for i in range(n_classes):
            imageconvlist.append(tf.transpose(tf.reshape(imageconv1[i],[1,60,80,l1_outputchan]), [3,1,2,0]))        
            tf.summary.image('number_{}'.format(i), imageconvlist[i],max_outputs = l1_outputchan)
    
    with tf.name_scope("f2pass"):
        imageconvlist2 = []
        imagepool1 = tf.layers.max_pooling2d(inputs=imageconv1, pool_size=[2 , 2], strides=2)
        imageconv2 = tf.nn.relu(tf.nn.conv2d(imagepool1, filter2,strides=[1,1,1,1],padding="SAME"))
        for i in range(n_classes):
            imageconvlist2.append(tf.transpose(tf.reshape(imageconv2[i],[1,30,40,l2_outputchan]), [3,1,2,0]))
            tf.summary.image('number_{}'.format(i), imageconvlist2[i],max_outputs = l2_outputchan)
    
    # add desired tensors to collection for easy later restoring       
    tf.add_to_collection("prediction", prediction)
    tf.add_to_collection("optimizer", optimizer)
    tf.add_to_collection("acc_op", acc_op )
    tf.add_to_collection("conf_mat", conf_mat )
    tf.add_to_collection("cost", cost )
    tf.add_to_collection("x", x )
    tf.add_to_collection("y", y )
    tf.add_to_collection("layer_keep_holder", layer_keep_holder )
            
    return prediction, optimizer, acc_op, conf_mat, cost, x, y, layer_keep_holder
    
def train_neural_network(x_train, y_train, x_test, y_test,x_img,prediction, optimizer,
                         acc_op, conf_mat, cost, x, y, layer_keep_holder,log_folder):
    print('test set size: {}'.format(len(y_test)))
    print('train set size: {}'.format(len(y_train)))
    # number of cycles of feed forward and back propagation
    hm_epochs = 15
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5, )
    i = 0
    print('starting training')
    with tf.Session() as sess:   
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(log_folder, sess.graph) # for 1.0
        # generate 2 summary merges. one for use with test data, one for use with train data.
        # data uninfluenced by test or train data is added to the train summary.
        trainmerge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='train') + \
                                      tf.get_collection(tf.GraphKeys.SUMMARIES, scope='visualization_filter1') + \
                                      tf.get_collection(tf.GraphKeys.SUMMARIES, scope='weights'))
        testmerge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='test'))
        imagepassmerge = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='f1pass') + \
                                          tf.get_collection(tf.GraphKeys.SUMMARIES, scope='f2pass'))
        
        #if checkpoint file exists in log_folder, reload previous model
        if os.path.isfile( log_folder + '/checkpoint'):
            print('previous version found. continguing')
            saver.restore(sess,tf.train.latest_checkpoint(log_folder))
            #read checkpoint file and cast number at the end to int
            ckpt = tf.train.get_checkpoint_state(log_folder)
            i = int(str(ckpt).split('-')[-1][:-2])
           
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for start, end in zip(range(0, len(x_train), batch_size), range(batch_size, len(x_train)+1, batch_size)):
                if end > len(x_train):
                    end = len(x_train)
                batch_x = np.array(x_train[start:end])
                batch_y = np.array(y_train[start:end])
                feed_dict = {x: batch_x, y: batch_y,layer_keep_holder: layer_keep}
                _, c = sess.run([optimizer, cost], feed_dict)
                epoch_loss += c
                i += end - start
                # generate test and train data summaries every 10000 datapoints
                if start % 10000 == 0:
                    feed_dict={x: x_test, y: y_test,layer_keep_holder: 1}
                    testsummary = sess.run(testmerge, feed_dict)
                    writer.add_summary(testsummary,i)
                    feed_dict={x: x_train[:testsize], y: y_train[:testsize],
                               layer_keep_holder: 1}
                    trainsummary = sess.run(trainmerge, feed_dict)
                    writer.add_summary(trainsummary,i)
            feed_dict={x: x_test, y: y_test, layer_keep_holder: 1}
            testacc, conf = sess.run([acc_op, conf_mat], feed_dict )
            print('Epoch ', epoch + 1, ' completed out of ', hm_epochs, ' ,epoch loss: ', epoch_loss)
            print('current test  accuracy: {}'.format(testacc))
            feed_dict={x: x_train[:testsize], y: y_train[:testsize],
                       layer_keep_holder: 1}
            trainacc = sess.run(acc_op , feed_dict)
            print('current train accuracy: {}'.format(trainacc))
            print(conf)
            feed_dict={x: x_img}
            imagepasssummary = sess.run(imagepassmerge, feed_dict)
            writer.add_summary(imagepasssummary,i)
            saver.save(sess, log_folder + '/canabalt_cnn', global_step=i)
#        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
#        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        accuracy = acc_op.eval({x: x_test, y: y_test, layer_keep_holder: 1})
        print('Final accuracy: ', accuracy  )
        saver.save(sess, log_folder + '/canabalt_cnn', global_step=i)
        writer.flush()
        writer.close()
    return accuracy

def split_data(data, testsize):            
    x = np.array([i[0] for i in data])
    y = np.array([i[1] for i in data])   
    y = cvt_y_to_onehot(y)
    y = y.reshape(-1,2)    
    x_train = x[:-testsize]
    x_test = x[-testsize:]
    y_train = y[:-testsize]
    y_test = y[-testsize:]
    print('data split')    
    return x_train, y_train, x_test, y_test

def load_and_split_data():
    data = np.load('training_data_balanced_tf_cnn_4l.npy')
    print('data loaded')
    return split_data(data,testsize)
  
def cvt_y_to_onehot(y):
    return np.array([[1,0] if i==0 else [0,1] for i in y])  

def pick_image_for_class(x_test,y_test, classes):
    x_images = []
    y_images = []   
    for i in range(classes):
        index = np.where(np.argmax(y_test,1)==i)[0][0]
        x_images.append( x_test[index])
        y_images.append( i)
    x_images = np.array(x_images)
    return x_images, y_images  

def build_and_train(epsilon,beta1,beta2,learning_rate,log_folder,
                    x_train, y_train, x_test, y_test,x_img):
    prediction, optimizer, acc_op, conf_mat, cost, x, y, \
    layer_keep_holder = create_graph(epsilon, beta1, beta2, learning_rate,
                                      filtersize, l1_outputchan,l2_outputchan,
                                      finallayer_in, denselayernodes,
                                      n_classes,log_folder)
        
    accuracy = train_neural_network(x_train, y_train, x_test, y_test,x_img,
                                    prediction, optimizer, acc_op, conf_mat,
                                    cost,x, y, layer_keep_holder,log_folder)
    return accuracy    

def gridsearch(log_folder,x_train, y_train, x_test,y_test,x_img):
    epsilons = [0.3, 1, 3]
    learning_rates = [0.003, 0.01]
    beta1s = [0.8, 0.9, 0.95]
    beta2s = [0.9, 0.95, 0.999]
    base_folder = "./logs/cnngpu_logs"
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
                                                   x_train, y_train, x_test,
                                                   y_test,x_img)
                        
                        results[add_dir] = accuracy
    
    print(results)
    return results

def main():
    x_train, y_train, x_test, y_test = load_and_split_data()   
    x_img, y_img = pick_image_for_class(x_test,y_test,n_classes)
    
    # set model parameters.
    # best model params found with gridsearch so far:
    # ep:0.3, lr: 0.003, b1: 0.9, b2: 0.95
    epsilon = 0.3
    learning_rate = 0.003
    beta1 = 0.9
    beta2 = 0.95
    log_folder = "./logs/cnnopparam_logs"
    
    accuracy = build_and_train(epsilon,beta1,beta2,learning_rate,log_folder,
                               x_train, y_train, x_test,y_test,x_img)    
    print(accuracy)
    return accuracy
     

if __name__ == "__main__":
    main()