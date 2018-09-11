#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 09:54:17 2018

@author: suliang

重点：

1. 基本逻辑：
    只要跟tensor相关的变量定义，运算定义，都需要用tf自带的，所以需要加tf.
    只要带tf的定义，都要在session里边才能运行
2. 张量类型：
    * tf.int32 表示32位有符号整数
    * tf.float32 表示32位有符号浮点数
    * tf.string 表示可变长度的字节数组？？？
    
2. 常量定义：通常用法是跟tf.Variable()一起使用，把常量赋值给变量
    * tc = tf.constant(5, [2,3]) 表示2行3列的常数5
    * tc = tf.ones([2,3], tf.int32) 表示全1张量，大小2x3
    * tc = tf.zeros([2,3], tf.int32) 表示全0张量，大小2x3
    * tc = 

3. 特殊变量定义：
    * ph1 = tf.placeholder() 表示一个带输入的占位函数
    * v1 = tf.Variable(2.0, name = 'v1') 表示定义一个值为2.0名字v1的变量
    * g1 = tf.get_variable(name = 'g1',  表示
                           shape = ,     表示
                           detype = , 
                           initializer=tf.constant_initializer(1))
    * 
    
3. 基本数学运算定义：
    * tf.multiply()


"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle


# 数据集生成 
def generate(sample_size, mean, cov, diff,regression):   
    num_classes = 2 
    samples_per_class = int(sample_size/2)

    X0 = np.random.multivariate_normal(mean, cov, samples_per_class)
    Y0 = np.zeros(samples_per_class)
    
    for ci, d in enumerate(diff):
        X1 = np.random.multivariate_normal(mean+d, cov, samples_per_class)
        Y1 = (ci+1)*np.ones(samples_per_class)
    
        X0 = np.concatenate((X0,X1))
        Y0 = np.concatenate((Y0,Y1))
        
    if regression==False: #one-hot  0 into the vector "1 0
        class_ind = [Y==class_number for class_number in range(num_classes)]
        Y = np.asarray(np.hstack(class_ind), dtype=np.float32)
    X, Y = shuffle(X0, Y0)
    
    return X,Y 


'''
# 创建变量
input_features = tf.placeholder(tf.float32, [None, input_dim])
input_lables = tf.placeholder(tf.float32, [None, lab_dim])

W = tf.Variable(tf.random_normal([input_dim,lab_dim]), name="weight")
b = tf.Variable(tf.zeros([lab_dim]), name="bias")

# 创建
output =tf.nn.sigmoid( tf.matmul(input_features, W) + b)
cross_entropy = -(input_lables * tf.log(output) + (1 - input_lables) * tf.log(1 - output))
ser= tf.square(input_lables - output)
loss = tf.reduce_mean(cross_entropy)
err = tf.reduce_mean(ser)
optimizer = tf.train.AdamOptimizer(0.04) #尽量用这个--收敛快，会动态调节梯度
train = optimizer.minimize(loss)  # let the optimizer train

maxEpochs = 50
minibatchSize = 25


# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(maxEpochs):
        sumerr=0
        for i in range(np.int32(len(Y)/minibatchSize)):
            x1 = X[i*minibatchSize:(i+1)*minibatchSize,:]
            y1 = np.reshape(Y[i*minibatchSize:(i+1)*minibatchSize],[-1,1])
            tf.reshape(y1,[-1,1])
            _,lossval, outputval,errval = sess.run([train,loss,output,err], feed_dict={input_features: x1, input_lables:y1})
            sumerr =sumerr+errval

        print ("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(lossval),"err=",sumerr/minibatchSize)
        
    train_X, train_Y = generate(100, mean, cov, [3.0],True)
    colors = ['r' if l == 0 else 'b' for l in train_Y[:]]
    plt.scatter(train_X[:,0], train_X[:,1], c=colors)
    #plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y)
    #plt.colorbar()
'''


if __name__ == '__main__':
    
    test_id = 1
    
    if test_id == 0:
        cov = np.eye(2)
        mean = np.random.randn(2)
        X,y = generate(1000, mean, cov, [3.0], True)
        plt.scatter(X[:,0],X[:,1], c = y*40+30)
    
    
    if test_id == 1:   # 测试一下placeholder， 相当于一个函数变量
        ph1 = tf.placeholder(tf.float32)
        ph2 = tf.placeholder(tf.float32)
        
        result = tf.multiply(ph1, ph2)
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            print(sess.run(result))
        
        
        
        
        
        