#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 18:49:49 2017

@author: aditya
"""

import tensorflow as tf
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)

# This is to activate tensorboard.


with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(x,feed_dict=None, options=None, run_metadata=None))
writer.close()

aa = tf.constant([2,2],name="a")
bb = tf.constant([3,6],name="b")
xx = tf.add(aa, bb,name="add")

# This is to activate tensorboard.

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    # tf.Session.run(fetches, feed_dict=None, options=None, run_metadata=None)
    print(sess.run(xx,feed_dict=None, options=None, run_metadata=None))
writer.close()

# tf​.​constant​(​value​,​ dtype​=​None​,​ shape​=​None​,​ name​=​'Const'​,​ verify_shape​=​False)
# constant of 1d tensor (vector)
a = tf.constant([2, 2], name="vector")

# constant of 2x2 tensor (matrix)
b = tf.constant([[0, 1], [2, 3]], name="b")


#tf​.​zeros​(​shape​,​ dtype​=​tf​.​float32​,​ name​=​None)
# create a tensor of shape and all elements are zeros
tf.zeros([2, 3], tf.int32)

tf.zeros_like(a) 

tf.ones([2, 3], tf.int32) 

#==> [[1, 1, 1], [1, 1, 1]]


#tf​.​fill​(​dims​,​ value​,​ name​=​None​)

tf.fill([2, 3], 8) 

#==> [[8, 8, 8], [8, 8, 8]]


tf.linspace(10.0, 13.0, 4, name="linspace") 
#==> [10.0 11.0 12.0 13.0]

# 'start' is 3, 'limit' is 18, 'delta' is 3
tf.range(3, 18, 3) 
#==> [3, 6, 9, 12, 15]
# 'start' is 3, 'limit' is 1, 'delta' is -0.5
tf.range(3, 1, 0.5) 
#==> [3, 2.5, 2, 1.5]
# 'limit' is 5
tf.range(5) 
#==> [0, 1, 2, 3, 4]


#tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
#tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None,
#name=None)
#tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None,
#name=None)
#tf.random_shuffle(value, seed=None, name=None)
#tf.random_crop(value, size, seed=None, name=None)
#tf.multinomial(logits, num_samples, seed=None, name=None)
#tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)
