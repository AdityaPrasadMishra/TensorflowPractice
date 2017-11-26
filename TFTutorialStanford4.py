#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 01:26:40 2017

@author: aditya
"""
#STill doing Variables
# create a variable whose original value is 2
import tensorflow as tf

a = tf.Variable(2, name="scalar")
# assign a * 2 to a and call that op a_times_two
a_times_two = a.assign(a * 2)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
     # have to initialize a, because a_times_two op depends on the value of a
    sess.run(a_times_two) # >> 4
    sess.run(a_times_two) # >> 8
    sess.run(a_times_two) # >> 16
    print(a.eval())
    
W = tf.Variable(10)
with tf.Session() as sess:
    sess.run(W.initializer)
    print(sess.run(W.assign_add(10))) # >> 20
    print(sess.run(W.assign_sub(2))) # >> 18
    
#Each Session can keep it's separate set of variable values
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(W.initializer)
sess2.run(W.initializer)
print(sess1.run(W.assign_add(10))) # >> 20
print(sess2.run(W.assign_sub(2))) # >> 8
print(sess1.run(W.assign_add(100))) # >> 120
print(sess2.run(W.assign_sub(50))) # >> -42
sess1.close()
sess2.close()

#One Varable is adependent on another variable    
W = tf.Variable(tf.truncated_normal([700, 10]))
U = tf.Variable(W * 2)

#make sure that W is initialized before its value is used to initialize U
U = tf.Variable(W.intialized_value() * 2)

