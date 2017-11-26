#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 01:40:15 2017

@author: aditya
"""

import tensorflow as tf
#Time to get Thrifty

#Interactive session

sess = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
# We can just use 'c.eval()' without passing 'sess'
print(c.eval())
sess.close()

#Sometimes, we will have two independent ops but you’d like to specify which op should be run first, 
# then you use tf.Graph.control_dependencies(control_inputs)

# your graph g have 5 ops: a, b, c, d, e
# with g.control_dependencies([a, b, c]):
 # `d` and `e` will only run after `a`, `b`, and `c` have executed.
 # d = ...
 # e = …
 
#Placeholder and feed_dict

#Phase 1: assemble a graph
#Phase 2: use a session to execute operations in the graph.
# For example, f(x, y) = x*2 + y.
#x, y are placeholders for the actual values
 # create a placeholder of type float 32-bit, shape is a vector of 3 elements
a = tf.placeholder(tf.float32, shape=[3])
# create a constant of type float 32-bit, shape is a vector of 3 elements
b = tf.constant([5, 5, 5], tf.float32)
# use the placeholder as you would a constant or a variable
c = a + b # Short for tf.add(a, b)
#If we try to fetch c, we will run into error.
with tf.Session() as sess:
    #print(sess.run(c))
    writer = tf.summary.FileWriter('./my_graph', sess.graph)
    print(sess.run(c, {a: [1, 2, 3]}))
writer.close()

# create Operations, Tensors, etc (using the default graph)
a = tf.add(2, 5)
b = tf.multiply(a, 3)
# To check if a tensor is feedable or not, use:
#print(tf.Graph.is_feedable(a))

# start up a `Session` using the default graph
sess = tf.Session()
# define a dictionary that says to replace the value of `a` with 15
replace_dict = {a: 15}
# Run the session, passing in `replace_dict` as the value to `feed_dict`
print(sess.run(b, feed_dict=replace_dict)) # returns 45
sess.close()

