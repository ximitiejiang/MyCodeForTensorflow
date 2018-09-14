#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 09:54:17 2018

@author: suliang

重点：tensorflow的基本语法

0. 运行方式
    * sess.run(2*x, feed_dict = {x:3})  # 运行一个运算，传递进去一个数值
    * sess.run([optimizer, cost], feed_dict={})  # 运行一个函数，传递进去一组数值

1. 基本逻辑：
    * 只要跟tensor相关的变量定义，运算定义，都需要用tf自带的，所以需要加tf.
    * 只要带tf的定义，都要在session里边才能运行
    * tf.get_variable在创建变量是会检查图上是否已创建，如果已创建并且本次未设置为共享，则报错
      解决办法是在程序尾增加一条tf.reset_default_graph()
      
2. 张量类型：
    * tf.int32 表示32位有符号整数
    * tf.float32 表示32位有符号浮点数
    * tf.string 表示可变长度的字节数组？？？
    
3. 常量定义：通常用法是跟tf.Variable()一起使用，把常量赋值给变量
    * tc = tf.constant(5, [2,3]) 表示2行3列的常数5
    * tc = tf.ones([2,3], tf.int32) 表示全1张量，大小2x3
    * tc = tf.zeros([2,3], tf.int32) 表示全0张量，大小2x3
    * tc = tf.random_normal([784, 10], mean=0,stddev=1.0) 表示正态分布随机数，shape为784x10, 2倍标准差之间

4. 特殊变量定义：
    * ph1 = tf.placeholder() 表示一个带输入的占位函数
    * v1 = tf.Variable(2.0, name = 'v1') 表示定义一个值为2.0名字v1的变量
    * g1 = tf.get_variable(name = 'g1',  表示
                           shape = ,     表示
                           detype = , 
                           initializer=tf.constant_initializer(1))
5. 基本形状变化：
    * tf.shape(lst)
    * tf.reshape(lst, [3,3]) # 把数据变为3x3
    * 
    
6. 基本数学运算定义：
    * tf.multiply(x,y, name = None)  表示按位置相乘
    * tf.add(x,y, name=None)
    * tf.pow(x,y, name=None)  表示幂次x^y
    * tf.matmul(x, w)   表示点积

7. 基本矩阵操作
    * tf.stack([tensor1, tensor2], axis=0) # 堆叠
    * tf.one_hot(indices,depth,on_value, off_value, axis) # 独热编码
    * 

8. 降维操作

9. 序列比较与索引提取
    * tf.argmin(data,axis=1)
    * tf.unique(data) # 返回元祖(lst, idx)，lst为唯一化列表，idx为对应y的index

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
            _,lossval, outputval,errval = sess.run([train,loss,output,err], 
                                                   feed_dict={input_features: x1, input_lables:y1})
            sumerr =sumerr+errval

        print ("Epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(lossval),"err=",sumerr/minibatchSize)
        
    train_X, train_Y = generate(100, mean, cov, [3.0],True)
    colors = ['r' if l == 0 else 'b' for l in train_Y[:]]  # 这个颜色定义逻辑好，不过就是比我的复杂
    plt.scatter(train_X[:,0], train_X[:,1], c=colors)
    #plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y)
    #plt.colorbar()



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
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(result))
        
        
        
        
        
        