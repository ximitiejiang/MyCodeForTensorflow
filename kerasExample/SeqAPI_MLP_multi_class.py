#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 13:39:47 2018

@author: suliang

用tensorflow/keras实现MLP多层感知机

重点：
1. 通过keras创建模型的两种方式
    * 通过sequential API: 序列式模型，也就是按序列逐层建立模型
    * 通过functional API: 函数是模型，也就是通过继承类来创建模型
    
2. 使用sequential API创建一个模型的过程
    * 导入
        - import sequential
        - import layers
        - import optimizers
        
    * 新建sequential模型
    * 加入层
    * 模型编译
    * 模型拟合
    * 模型评分

3. 区分各种layers: 每一层相当于一个函数，用来执行一种运算。
   比如dense层执行输入与参数的点积加上偏置。
   比如dropout层
   比如
    * 第一层一定要额外传入一个参数：input shape = (,)
    * Dense(units,   # 神经元个数
            activation=None, 
            use_bias=True, 
            kernel_initializer='glorot_uniform', 
            bias_initializer='zeros', 
            kernel_regularizer=None, 
            bias_regularizer=None, 
            activity_regularizer=None, 
            kernel_constraint=None, 
            bias_constraint=None)
    * Dropout(rate, 
              noise_shape=None, 
              seed=None)

4. 区分各种activation: 即各种激活函数，包括'relu', 'softmax', 'tanh'
    * 'relu'
    * 'softmax'
    * 'sigmoid'
    * 'tanh'
    * 'linear'

4. 区分各种optimizers: 即各种参数计算算法，包括梯度下降等
    * SGD(lr=0.01, 
          momentum=0.0, 
          decay=0.0, 
          nesterov=False)
    * RMSprop(lr=0.001, 
              rho=0.9, 
              epsilon=None, 
              decay=0.0)
    * Adagrad(lr=0.01, 
              epsilon=None, 
              decay=0.0)
    * Adam(lr=0.001, 
           beta_1=0.9, 
           beta_2=0.999, 
           epsilon=None, 
           decay=0.0, 
           amsgrad=False)

5. 区分各种loss: 即各种损失函数，包括交叉熵损失，mse损失
    * mean_squared_error(y_true, y_pred)  # 均方误差
    * categorical_crossentropy(y_true, y_pred) # 交叉熵

6. 区分各种metrics
    * 

7. 区分model fit时的epochs和batch_size
    * epochs: 一次性用于训练模型的样本总数，每一个epoch可以训练出一个模型得到一个评分
    * batch_size: 每个梯度更新的样本数量，如果未指定，keras默认是32

8. 



"""
# 以下在没有打开eager execution时已运行通过

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


def createSeqModel_1(): #seq model创建技巧1, 先创建模型再往模型内添加层
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=20))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    return model
    

def createSeqModel_2():  # seq model创建技巧2, 一次性创建整个模型和各个层
    model = Sequential([Dense(64， activatin = 'relu', input_dim = 20),
                        Dropout(),
                        Dense(),
                        Dropout(),
                        Dense(10, activation='softmax')])
    return model


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

