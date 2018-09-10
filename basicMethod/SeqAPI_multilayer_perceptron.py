#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 13:39:47 2018

@author: suliang

用tensorflow/keras实现MLP多层感知机

"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 20))
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

# 创建sequantial模型
model = Sequential()

# 添加dense层(64个节点) - 激活函数relu，由于是第一层，需要指定输入数据维度
# 输入特征是20列特征，所以指定input_dim = 20
model.add(Dense(64, activation='relu', input_dim=20))
# 添加dropout层
model.add(Dropout(0.5))
# 添加dense层 - 激活函数relu
model.add(Dense(64, activation='relu'))
# 添加dropout层 
model.add(Dropout(0.5))
# 添加dense层 - 激活函数softmax用于多分类
model.add(Dense(10, activation='softmax'))

# 定义optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# 模型编译
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
# 模型拟合
model.fit(x_train, y_train,
          epochs=20,
          batch_size=128)
# 模型评分
score = model.evaluate(x_test, y_test, batch_size=128)

