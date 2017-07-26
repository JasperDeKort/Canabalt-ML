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


#train_data = np.load('training_data 102000.npy')
#train_data = np.concatenate((train_data2, train_data),axis = 0)

#for data in train_data:
#    img = data[0]
#    choice = data[1]
#    cv2.imshow('test', img)
#    print(choice)
#    if cv2.waitKey(15) & 0xFF == ord('q'):
#        cv2.destroyAllWindows()
#        break

def mergepreviousimages(train_data):
    new_train_data = [[np.append(train_data[i][0],[ train_data[i-3][0]]),train_data[i][1]] for i in range(len(train_data))]
    print(new_train_data[1][0].shape)
    return new_train_data[3:-3]

def balance_data(train_data, overwrite=False):  
    spaces = []
    nones = []
    
    train_data = mergepreviousimages(train_data)
    shuffle(train_data)    
    
    for data in train_data:
        if data[1] == 0:
            nones.append(data)
        elif data[1] == 1:
            spaces.append(data)
        else:
            print('error value at {}'.format(data.index))
    
    print('spaces: {}'.format(len(spaces)))
    fulldata = spaces + nones[:int(len(spaces)*2)]
    shuffle(fulldata)
    np.save('training_data_balanced_tf.npy',fulldata)
    df = pd.DataFrame(fulldata)
    print(Counter(df[1].apply(str)))
    
    return fulldata

def main():
    train_data = np.load('training_data.npy')
    balance_data(train_data)
    
if __name__ == "__main__":
    main()
