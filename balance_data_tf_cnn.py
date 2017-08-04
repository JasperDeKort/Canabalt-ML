# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 17:39:56 2017

@author: jaspe
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:40:25 2017

@author: jaspe
"""

import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

def rebuild_2d(train_data):
    newdata = []
    for sample in train_data:
        newdata.append([sample[0].reshape(60,80),sample[1]])        
    return newdata   

#def merge_previous_images(train_data):
#    new_train_data = [[np.stack([train_data[i][0],train_data[i-3][0]], 2),train_data[i][1]] for i in range(len(train_data))]
#    print(new_train_data[0][0].shape)
#    return new_train_data[3:-3]

def merge_previous_images(train_data):
    new_train_data = [[np.stack([train_data[i][0]], 2),train_data[i][1]] for i in range(len(train_data))]
    print(new_train_data[0][0].shape)
    return new_train_data[3:-3]

def balance_data(train_data, overwrite=False):
    spaces = []
    nones = []
    
    train_data = merge_previous_images(train_data)
    shuffle(train_data)    
    
    for data in train_data:
        if data[1] == 0:
            nones.append(data)
        elif data[1] == 1:
            spaces.append(data)
        else:
            print('error value at {}'.format(data.index))
    
    print('spaces: {}'.format(len(spaces)))
    fulldata = spaces + nones[:int(len(spaces))]
    shuffle(fulldata)
    np.save('training_data_balanced_tf_cnn.npy',fulldata)
    df = pd.DataFrame(fulldata)
    print(Counter(df[1].apply(str)))
    
    return fulldata

def main():
    train_data = np.load('training_data.npy')
    print('data loaded')
    train_data = rebuild_2d(train_data)
    balance_data(train_data)
    
if __name__ == "__main__":
    main()
