#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 02:00:14 2017

@author: aditya
"""
import tensorflow as tf
#Normal loading:

x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
z = tf.add(x, y) # you create the node for add node before executing the graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./my_graph/l2', sess.graph)
    for _ in range(10):
        print(sess.run(z))
    writer.close()
    
# Lazy loading:
x = tf.Variable(10, name='x')
y = tf.Variable(20, name='y')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./my_graph/l2', sess.graph)
    for _ in range(10):
        print(sess.run(tf.add(x, y))) # someone decides to be clever to save one line of code
    writer.close()
