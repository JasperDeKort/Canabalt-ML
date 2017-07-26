# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 17:50:55 2017

@author: jaspe
"""

from sklearn.neural_network import MLPClassifier
#from sklearn import model_selection
from sklearn.metrics import confusion_matrix
import numpy as np
import pickle
from collections import Counter
import os

def train(mlp = None,train_data=None , iteration=''):
    if train_data==None:
        train_data = np.load('training_data_balanced.npy')
        
    X = [i[0] for i in train_data]
    y = [i[1] for i in train_data]
    
    X_train = X[:-2000]
    X_test = X[-2000:]
    
    y_train = y[:-2000]
    y_test = y[-2000:]
    
    if mlp == None:
        mlp = MLPClassifier(solver='adam', alpha=1e-4,hidden_layer_sizes=(200,50), warm_start=True)
        print('starting training of new network')
    else:
        print('previous network passed, using hotstart')
    
    mlp.fit(X_train, y_train)
    print('training done')
    
    confidence = mlp.score(X_test, y_test)   
    print('expected accuracy is: {}'.format(confidence))
    
    predictions = mlp.predict(X)
    print('actual spread: ', Counter(y))
    print("predicted spread: ", Counter(predictions))    
    print(confusion_matrix(y, predictions))  
    with open('MLPtrained_{}.pickle'.format(iteration),'wb') as f:
        pickle.dump(mlp,f)
    return mlp

def main():
    with open('iternumber.pickle','rb') as f:
        iternumber = pickle.load(f)
    with open('MLPtrained_{}.pickle'.format(iternumber - 1 ),'rb') as f:
        mlp = pickle.load(f) 
    train(mlp=None, iteration=str(1))

        
if __name__ == "__main__":
    main()


