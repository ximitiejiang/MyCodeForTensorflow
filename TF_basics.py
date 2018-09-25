#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 14:21:50 2018

@author: suliang
"""

import tensorflow as tf

'''
创建常量
'''
tc1 = tf.constant(5, [2,3])  # 表示2行3列的常数5
tc2 = tf.ones([2,3], tf.int32) # 表示全1张量，大小2x3
tc3 = tf.zeros([2,3], tf.int32) # 表示全0张量，大小2x3
tc4 = tf.random_normal([784, 10], # mean=0,stddev=1.0)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    tf.reset_default_graph()
    sess.run(print(tc))


'''
创建变量
'''
tp1 = tf.placeholder() # 表示一个带输入的占位函数
tv1 = tf.Variable(2.0, name = 'v1') # 表示定义一个值为2.0名字v1的变量
tg1 = tf.get_variable(name = 'g1',
                      shape = (2,3)
                      detype = int32, 
                      initializer=tf.constant_initializer(1))
