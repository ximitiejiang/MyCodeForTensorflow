#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:00:07 2018

@author: suliang

重点：

1. 激活函数的概念
    * sigmoid函数: tf.nn.sigmoid(x, name=None)
    * tanh函数: tf.nn.tanh(x, name=None)
    * ReLU函数: tf.nn.relu(x, name=None), 常规relu是以0为阀值
               tf.nn.relu6(x,name=None), 这个relu是以6为阀值，为了防止梯度爆炸
    * softmax函数：tf.nn.softmax(logits, name=None)
    * 其中ReLU函数是当下最好的激活函数，因为他处理后的函数有更好的稀疏性，数据转化为只有最大数值和0的稀疏矩阵
      (稀疏矩阵就是矩阵中0占大多数的矩阵，对应有稠密矩阵，对稀疏矩阵可以打分评价他的稀疏性即含0百分比)
      稀疏矩阵在计算机中可以简化存储空间，比如可以通过csr_matrix()函数获得一个简单的稀疏表示矩阵
      所以以稀疏性数据表达原特征，可使神经网络在迭代运算又快又好
    
    * Swish函数：google发现效果更优于ReLU的激活函数, 大部分模型只要把ReLU换成Swish准确率都能提高
      def Swish(x, beta = 1):
          return x*tf.nn.sigmoid(x*beta)  
    
2. 损失函数的概念 (其中logits代表标签值，outputs代表预测值)
    * 均方差
        - 通常自己写：mse = tf.reduce_mean(tf.square(tf.sub(logits, outputs)))
        
    * 交叉熵: 包括了sigmoid交叉熵，softmax交叉熵，sparse交叉熵，加权sigmoid交叉熵
        - tf.nn.sigmoid_cross_entropy_with_logits(logits, targets)
        - tf.nn.softmax_cross_entropy_with_logits(logits, targets)
        - 可以自己写：-tf.reduce_sum(labels*tf.log(logits),1)
        
    * 损失函数的选择建议：
        - 如果是sigmoid做激活函数，建议用交叉熵
        - 如果是softmax做激活函数，建议用交叉熵
        - 如果是reLU或其他激活函数，建议用MSE

3. 优化器optimizer
    * 优化器就是梯度下降算法，在tensorflow带有5-6个优化器
        - 常规梯度下降：optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        - Adam算法： optimizer = tf.train.AdamOptimizer(learning_rate=0.001,beta1=0.9)
    * 比较常用的优化器是Adam优化器.
    * 优化方法是：
        -      

4. 学习率alpha
    * 学习率alpha是用于梯度下降时的步长设置
    * 常见问题是：学习率过大，训练速度会提高，但精度会下降；学习率过小，精度会提升，但太耗费时间
    * 设置alpha的方法是：采用退化学习率，即学习率衰减，开始时学习率大，后期学习率逐渐减小
    * 增加batch_size跟减小学习率，效果是一样的。
    * 自定义学习率衰减函数：
      def exponential_decay(learning rate, global_step, decay_step, decay_rate)

5. 几个size的区别：
    * epoch: 代，一个数据集就是一个epoch，或者整个数据集轮转次数就是多少个epoch。
    * batch size：把整个数据集拆分成多个batch，相当于每次送入梯度下降时的数据包大小
    * iteration: 迭代次数,即batch个数，或者参数更新次数
    
    * 设立batch size的目的：之所以不是整个数据集一次性送进神经网络，是因为数据集太大会比较费时。
    * 设立batch size的原则：
        - 基于允许可用的内存大小定义
        - 考虑数据的平衡性，尽量让batch中包含所有样本类别
        - 观察loss/metrics，如果不稳定则增加batch size或减小学习率，如果过于稳定则减小batch size
    
    * 设立epoch的目的：多次训练，可以更好的收敛，虽然是同一个数据集，但因为参数更新过了，也会有效果
    * 设立epoch的原则：
        - 10次epoch可以看一下损失下降的情况
        - 
    
    * 一个计算示例：CIFAR数据集，50000张训练图片
        - 先定义一个epoch = 50000张，每个epoch跑完整个数据集
        - 再定义batch_size = 256，即每次用256张图片跑一个梯度
        - 再定义iteration = training set size/batch_size + 1 = 196个，
          即每一个batch_size跑一次梯度，更新一次权重参数
        

梳理tensorflow的神经网络运行过程：
    * 神经网络节点定义------------------这条需要设置w,b的shape
    * 数据划分尺寸定义------------------这条需要设置epoch和batch_size
    * 激活函数定义---------------------这条需要设置或者自定义激活函数
    * 损失函数定义---------------------这条需要设置或者自定义损失函数
    * 权值初始化-----------------------tensorflow集成
    * 前向传播计算----------------------tensorflow集成
    * 梯度下降求解：先求激活函数导数，再求残差，最后求梯度
      --------------------------------这条需要设置optimizer优化器
    * 权重更新：基于梯度下降算法更新-------tensorflow集成

"""
import tensorflow as tf

labels = [[0,0,1],
          [0,1,0]]
logits = [[2,0.5, 6],
          [0.1,0,3]]

logits_scaled = tf.nn.softmax(logits)
logits_scaled2 = tf.nn.softmax(logits_scaled)

with tf.Session() as sess:
    print('scaled = ', sess.run(logits_scaled))
    print('scaled = ', sess.run(logits_scaled2))



