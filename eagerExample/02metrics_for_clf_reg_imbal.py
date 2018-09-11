#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 20:59:31 2018

@author: suliang

基于eager，用tensorflow创建模型解决如下问题：
    * 多分类
    * 回归
    * 数据分布不平衡

"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from sklearn.datasets import load_wine
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA


tfe.enable_eager_execution()

# 导入数据
wine_data = load_wine()

# 特征工程1: 对数据做标准化(做去中心化是PCA的前置步骤，这里用标准化替代去中心化也达到均值为0的效果)
wine_data.data = (wine_data.data - np.mean(wine_data.data, axis=0))/ \
                  np.std(wine_data.data, axis=0)
# 特征工程2: 对数据PCA降维成2为特征
X_pca = PCA(n_components=2, random_state=2018).fit_transform(wine_data.data)


# 创建神经网络
class two_layer_nn(tf.keras.Model):
    def __init__(self, output_size=2, loss_type='cross-entropy'):
        super(two_layer_nn, self).__init__()
        """ Define here the layers used during the forward-pass 
            of the neural network.     
            Args:
                output_size: int (default=2). 
                loss_type: string, 'cross-entropy' or 'regression' (default='cross-entropy')
        """   
        # 第一层隐藏层
        self.dense_1 = tf.layers.Dense(20, activation=tf.nn.relu)
        # 第二层隐藏层
        self.dense_2 = tf.layers.Dense(10, activation=tf.nn.relu)
        # 输出层. Unscaled log probabilities
        self.dense_out = tf.layers.Dense(output_size, activation=None)     
        # 初始化损失类型
        self.loss_type = loss_type
    
    def predict(self, input_data):
        """ 计算前向神经网络各层    
            Args:
                input_data: 2D tensor of shape (n_samples, n_features).   
            Returns:
                logits: unnormalized predictions.
        """
        layer_1 = self.dense_1(input_data)
        layer_2 = self.dense_2(layer_1)
        logits = self.dense_out(layer_2)
        return logits
    
    def loss_fn(self, input_data, target):
        """ 定义训练中使用的损失函数         
        """
        preds = self.predict(input_data)
        if self.loss_type=='cross-entropy':
            loss = tf.losses.sparse_softmax_cross_entropy(labels=target, logits=preds)
        else:
            loss = tf.losses.mean_squared_error(target, preds)
        return loss
    
    def grads_fn(self, input_data, target):
        """ Dynamically computes the gradients of the loss value
            with respect to the parameters of the model, in each
            forward pass.
        """
        with tfe.GradientTape() as tape:
            loss = self.loss_fn(input_data, target)
        return tape.gradient(loss, self.variables)
    
    def fit(self, input_data, target, optimizer, num_epochs=500, 
            verbose=50, track_accuracy=True):
        """ 训练模型, 使用已选的优化器和需要的循环次数. 也存储了每个循环之后的模型精度
        """   
        
        if track_accuracy:
            # Initialize list to store the accuracy of the model
            self.hist_accuracy = []     
            # Initialize class to compute the accuracy metric
            accuracy = tfe.metrics.Accuracy()

        for i in range(num_epochs):
            # Take a step of gradient descent
            grads = self.grads_fn(input_data, target)
            optimizer.apply_gradients(zip(grads, self.variables))
            if track_accuracy:
                # Predict targets after taking a step of gradient descent
                logits = self.predict(X)
                preds = tf.argmax(logits, axis=1)
                # Compute the accuracy
                accuracy(preds, target)
                # Get the actual result and add it to our list
                self.hist_accuracy.append(accuracy.result())
                # Reset accuracy value (we don't want to track the running mean accuracy)
                accuracy.init_variables()

# 定义输入数据和标签数据
X = tf.constant(wine_data.data)
y = tf.constant(wine_data.target)
# 定义优化器
optimizer = tf.train.GradientDescentOptimizer(5e-1)
# 定义模型
model = two_layer_nn(output_size=3)
# 定义循环次数
num_epochs = 5
# 模型拟合
model.fit(X, y, optimizer, num_epochs=num_epochs)

# 获得模型预测值
logits = model.predict(X)
preds = tf.argmax(logits, axis=1)



