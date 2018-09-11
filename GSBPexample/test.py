#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 13:45:12 2018

@author: suliang
"""

import numpy as np
import tensorflow as tf

ph1 = tf.placeholder(tf.float32)
ph2 = tf.placeholder(tf.float32)
        
result = tf.multiply(ph1, ph2)
        
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(result))