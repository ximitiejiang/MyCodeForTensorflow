#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 17:05:39 2018

@author: suliang
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras

# 以下的运算都可以在session之外直接运行，比以前更方便检查程序
# 缺点是在spyder的变量窗口看不到变量值（只能在unsupport data type里边看到变量名）
# 但可以通过print()看到
x = tf.matmul([[1, 2],[3, 4]],[[4, 5],[6, 7]])
y = tf.add(x, 1)
z = tf.random_uniform([5, 3])
print(x)
print(y)
print(z)


# 以下展示如何与numpy的进行无缝切换
np_x = np.array(2., dtype=np.float32)
x = tf.constant(np_x)

py_y = 3.
y = tf.constant(py_y)

z = x + y + 1

print(z)
print(z.numpy())  # 实现与numpy的无缝切换：用.numpy()把tensor格式转变为array格式


'''构建模型的3种方式
   1. 通过layers来逐层构建模型
   2. 通过
   3. 编写keras的子类来构建模型

'''
import tensorflow as tf
from tensorflow import keras

# 方式1： 建立简单模型，用Sequential API
model = keras.Sequential()
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

layers.Dense(64, activation='sigmoid')
# Or:
layers.Dense(64, activation=tf.sigmoid)

# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
layers.Dense(64, kernel_regularizer=keras.regularizers.l1(0.01))
# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
layers.Dense(64, bias_regularizer=keras.regularizers.l2(0.01))

# A linear layer with a kernel initialized to a random orthogonal matrix:
layers.Dense(64, kernel_initializer='orthogonal')
# A linear layer with a bias vector initialized to 2.0s:
layers.Dense(64, bias_initializer=keras.initializers.constant(2.0))


# 方式2： 建立较复杂模型，用Function API - 这是比较推荐的方法
inputs = keras.Input(shape=(32,))
x = keras.layers.Dense(64, activation='relu')(inputs)
x = keras.layers.Dense(64, activation='relu')(x)
predictions = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=predictions)
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels, batch_size=32, epochs=5)

# 方式3：建立较复杂模型，用subclass模型子类 （更灵活，更复杂，也更容易出错）




