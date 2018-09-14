#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:44:30 2018

@author: suliang

重点：
1. 调用MNIST数据集: 数据里边有3大块（train, test, validation）
    * mnist.train.images  # 张量，放置55000张图片，55000 x 784, 0代表黑色，1-255代表其他像素
    * mnist.test.images  # 张量，放置10000张图片
    * mnist.validation.images  # 张量，放置5000张图片
    
    * mnist.train.labels  # 介于0-9之间以one-hot编码表示，为10位的特征。0表示为[1,0,0,0,0,0,0,0,0,0]
    * mnist.test.labels
    * mnist.validation.labels

2. 神经网络tf.matmul(x, W) + b的shape分析
    * x是输入，由输入数据定义，x.shape得到
    * b是偏置，加上他就是输出了，所以可以认为跟输出同shape，由输出定义
    * w是参数，连接输入输出，w的shape = (n_x, n_b)

"""

# 以下为一个无隐藏层的单层感知机，并配以softmax激活函数，也就等价于softmax分类器了
# 无隐藏层单层感知机做分类，精度可达到 0.8342

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

path = '/Users/suliang/MyDatasets/MNIST/'
mnist = input_data.read_data_sets(path, one_hot=True)

import pylab 

tf.reset_default_graph()  # 有get_variable(),所以用这句重置图的所有变量

x = tf.placeholder(tf.float32, [None, 784]) # mnist data维度 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 数字=> 10 classes

# Set model weights
W = tf.Variable(tf.random_normal([784, 10]))  # 没有隐藏层的单层感知机，w的维度m=输入层节点数，n=输出层节点数
b = tf.Variable(tf.zeros([10]))

# 构建模型
pred = tf.nn.softmax(tf.matmul(x, W) + b) # 输出层为Softmax函数作为激活函数

# 对于输出层的损失函数，如果是多分类的softmax，则用交叉熵做损失函数
# 如果是reLU，则用mse做损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

#参数设置
learning_rate = 0.01
# 使用梯度下降优化器
# 先定义优化器传入learning_rate, 然后minimize()函数传入损失函数
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

training_epochs = 25  # 训练25轮
batch_size = 100     # 每次用100个样本更新参数
display_step = 1
saver = tf.train.Saver()
model_path = "log/521model.ckpt"

# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())# 变量初始化

    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 遍历全部数据集
        for i in range(total_batch):
            # 借用mnist数据集里自带的next_batch()函数生成batch_x, batch_y
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 运行优化器和cost函数，占位符参数传递进去
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # 显示训练中的详细信息
        if (epoch+1) % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print( " Finished!")

    # 测试 model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    # Save model weights to disk
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)