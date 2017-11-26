#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 01:09:05 2017

@author: aditya
"""

#Variables are fun aren't they
#x.initializer # init
#x.value() # read op
#x.assign(...) # write op
#x.assign_add(...)
## and more

import tensorflow as tf
my_const = tf.constant([1.0, 2.0], name="my_const")
print(tf.get_default_graph().as_graph_def());

#create variable a with scalar value
a = tf.Variable(2, name="scalar")
#create variable b as a vector
b = tf.Variable([2, 3], name="vector")

#create variable c as a 2x2 matrix
c = tf.Variable([[0, 1], [2, 3]], name="matrix")
# create variable W as 784 x 10 tensor, filled with zeros
W = tf.Variable(tf.zeros([784,10]))

#The easiest way is initializing all variables at once using: tf.global_variables_initializer()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

#Only Initialize a and b
init_ab = tf.variables_initializer([a, b], name="init_ab")
with tf.Session() as sess:
    sess.run(init_ab)

#Initialize W
with tf.Session() as sess:
    sess.run(W.initializer)
    print(W)

W = tf.Variable(tf.truncated_normal([700, 10]))
with tf.Session() as sess:
    sess.run(W.initializer)
    #gives value to variables
    print(W.eval())

W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
    sess.run(assign_op)
    print(W.eval()) # >> 100