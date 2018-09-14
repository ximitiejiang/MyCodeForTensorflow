#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 12:21:41 2018

@author: suliang

一个重要的用于学习率衰减的函数：
lr = tf.train.exponential_decay(initial_lr,
                                global_step = global_step,
                                decay_steps = 10,
                                decay_rate=0.9)

"""


import tensorflow as tf
global_step = tf.Variable(0, trainable=False)
initial_lr = 0.1
lr = tf.train.exponential_decay(initial_lr,
                                global_step = global_step,
                                decay_steps = 10,
                                decay_rate=0.9)
opt = tf.train.GradientDescentOptimizer(learning_rate)
add_global = global_step.assign_add(1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print(sess.run(lr))
    
    for i in range(20):
        g, rate = sess.run([add_global, lr])
        
        print(g, rate)