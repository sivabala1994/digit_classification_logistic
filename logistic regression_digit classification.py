# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 12:57:38 2017

@author: sivakumar
"""
import matplotlib.pyplot as plt;
import numpy as np
from mnist import MNIST
def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, labels_test = map(np.array, mndata.load_testing())
    X_train = X_train/255.0
    X_test = X_test/255.0
    return X_train, X_test, labels_train,labels_test
X_train, X_test, labels_train,labels_test=load_dataset()


# part B: one hot encoding
y_train = np.zeros((X_train.shape[0], 10))
y_test=np.zeros((X_test.shape[0], 10))
y_train[np.arange(X_train.shape[0]), labels_train] = 1
y_test[np.arange(X_test.shape[0]), labels_test] = 1

        
#part C functions and accuracy calculations
def train(X, y,l):
    buf=np.dot(X.T,X)
    a=buf+l*np.eye(buf.shape[0])
    b=np.dot(X.T,y)
    W=np.linalg.solve(a,b)
    
    return W
W_train=train(X_train,y_train,10e-4)


def predict(W,X):
        y_label=np.zeros(X.shape[0])
        
        f=np.dot(W.T,X.T)
            
        y_label=np.argmax(f.T,axis=1)
        
        return y_label  
lab_test=predict(W_train,X_test)
lab_train=predict(W_train,X_train)
def accuracy(l1,l2):
    error = np.mean( l1 != l2 )
    acc=1-error
    return acc, error
test_acc,test_error=accuracy(labels_test,lab_test)
train_acc,train_error=accuracy(labels_train,lab_train)
#new_X_train=np.random.shuffle(X_train.flat)
#%%
# part d. training for transformation and cross validations for this particular dataset
def train_new(X, y,l):
    c=np.dot(X.T,X)+l*np.eye(X.shape[1])
    d=np.dot(X.T,y)
    W=np.linalg.solve(c,d)
    return W

train_x=X_train[:48000,:]
val_x=X_train[48000:60000,:]
train_y=y_train[:48000,:]
val_y=y_train[48000:60000,:]
train_lab=labels_train[:48000,]
val_lab=labels_train[48000:60000,]

def transform(g,x,b,p):
    x_new=np.zeros((p,x.shape[0]))
    for i in range(x.shape[0]):
        x_new[:,i]=np.cos((np.dot(g,x[i,:].T))+b)
    return x_new.T

def crossval(train_x,val_x):
    tr_err=np.zeros(6)
    val_err=np.zeros(6)
    for p in range(6):
       
        G=0.32*np.random.randn(p+1,784)
        b=np.random.uniform(0,2*3.14,p+1)
        x_new=transform(G,train_x,b,p+1)
        val_new=transform(G,val_x,b,p+1)
        
        w_new=train_new(x_new,train_y,10e-4)
        lab_test_val=predict(w_new,val_new)
        lab_train_val=predict(w_new,x_new)
        tr_acc,tr_err[p]=accuracy(train_lab,lab_train_val)
        val_acc,val_err[p]=accuracy(val_lab,lab_test_val)
        
    return tr_err,val_err

tr_err,val_err=crossval(train_x,val_x)
#%%
import matplotlib.pyplot as plt;
plt.plot(tr_err)
plt.plot(val_err)
plt.show()
