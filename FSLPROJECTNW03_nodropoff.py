#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 12:46:25 2017

@author: aditya
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 08:40:09 2017

@author: aditya
"""


import argparse
import sys
import tempfile
import math

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

FLAGS = None
MNIST_IMAGE_SIZE = 28
MNIST_IMAGE_PIXELS =28*28
OUTPUT_CLASSES = 10 
Batch_Size = 100
LEARNING_RATE = 0.01
hiddenlayer_units =128

def deepnnwithrelu(images):  
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, ). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9).  is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('hiddenlayer1'):
    weights = tf.Variable(
        tf.truncated_normal([MNIST_IMAGE_PIXELS, hiddenlayer_units],
                            stddev=1.0 / math.sqrt(float(MNIST_IMAGE_PIXELS))),
        name='weights')
    biases = tf.Variable(tf.zeros([hiddenlayer_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.name_scope('hiddenlayer2'):
    weights = tf.Variable(
        tf.truncated_normal([hiddenlayer_units, hiddenlayer_units],
                            stddev=1.0 / math.sqrt(float(hiddenlayer_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hiddenlayer_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
    
  with tf.name_scope('hiddenlayer3'):
    weights = tf.Variable(
        tf.truncated_normal([hiddenlayer_units, hiddenlayer_units],
                            stddev=1.0 / math.sqrt(float(hiddenlayer_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hiddenlayer_units]),
                         name='biases')
    hidden3 = tf.nn.relu(tf.matmul(hidden2, weights) + biases)
    
  with tf.name_scope('hiddenlayer4'):
    weights = tf.Variable(
        tf.truncated_normal([hiddenlayer_units, hiddenlayer_units],
                            stddev=1.0 / math.sqrt(float(hiddenlayer_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hiddenlayer_units]),
                         name='biases')
    hidden4 = tf.nn.relu(tf.matmul(hidden3, weights) + biases)
  
  with tf.name_scope('hiddenlayer5'):
    weights = tf.Variable(
        tf.truncated_normal([hiddenlayer_units, hiddenlayer_units],
                            stddev=1.0 / math.sqrt(float(hiddenlayer_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hiddenlayer_units]),
                         name='biases')
    hidden5 = tf.nn.relu(tf.matmul(hidden4, weights) + biases)
    
  with tf.name_scope('hiddenlayer6'):
    weights = tf.Variable(
        tf.truncated_normal([hiddenlayer_units, hiddenlayer_units],
                            stddev=1.0 / math.sqrt(float(hiddenlayer_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hiddenlayer_units]),
                         name='biases')
    hidden6 = tf.nn.relu(tf.matmul(hidden5, weights) + biases)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([hiddenlayer_units, 10])
    b_fc2 = bias_variable([10])

    y_output = tf.matmul(hidden6, W_fc2) + b_fc2
  return y_output 

def deepnnwithsigmoid(images):  
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, ). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9).  is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('hiddenlayer1'):
    weights = tf.Variable(
        tf.truncated_normal([MNIST_IMAGE_PIXELS, hiddenlayer_units],
                            stddev=1.0 / math.sqrt(float(MNIST_IMAGE_PIXELS))),
        name='weights')
    biases = tf.Variable(tf.zeros([hiddenlayer_units]),
                         name='biases')
    hidden1 = tf.nn.sigmoid(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.name_scope('hiddenlayer2'):
    weights = tf.Variable(
        tf.truncated_normal([hiddenlayer_units, hiddenlayer_units],
                            stddev=1.0 / math.sqrt(float(hiddenlayer_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hiddenlayer_units]),
                         name='biases')
    hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, weights) + biases)
    
  with tf.name_scope('hiddenlayer3'):
    weights = tf.Variable(
        tf.truncated_normal([hiddenlayer_units, hiddenlayer_units],
                            stddev=1.0 / math.sqrt(float(hiddenlayer_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hiddenlayer_units]),
                         name='biases')
    hidden3 = tf.nn.sigmoid(tf.matmul(hidden2, weights) + biases)
    
  with tf.name_scope('hiddenlayer4'):
    weights = tf.Variable(
        tf.truncated_normal([hiddenlayer_units, hiddenlayer_units],
                            stddev=1.0 / math.sqrt(float(hiddenlayer_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hiddenlayer_units]),
                         name='biases')
    hidden4 = tf.nn.sigmoid(tf.matmul(hidden3, weights) + biases)
  
  with tf.name_scope('hiddenlayer5'):
    weights = tf.Variable(
        tf.truncated_normal([hiddenlayer_units, hiddenlayer_units],
                            stddev=1.0 / math.sqrt(float(hiddenlayer_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hiddenlayer_units]),
                         name='biases')
    hidden5 = tf.nn.sigmoid(tf.matmul(hidden4, weights) + biases)
    
  with tf.name_scope('hiddenlayer6'):
    weights = tf.Variable(
        tf.truncated_normal([hiddenlayer_units, hiddenlayer_units],
                            stddev=1.0 / math.sqrt(float(hiddenlayer_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hiddenlayer_units]),
                         name='biases')
    hidden6 = tf.nn.sigmoid(tf.matmul(hidden5, weights) + biases)

  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([hiddenlayer_units, 10])
    b_fc2 = bias_variable([10])

    y_output = tf.matmul(hidden6, W_fc2) + b_fc2
  return y_output 

def deepnnwithelu(images):  
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, ). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9).  is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('hiddenlayer1'):
    weights = tf.Variable(
        tf.truncated_normal([MNIST_IMAGE_PIXELS, hiddenlayer_units],
                            stddev=1.0 / math.sqrt(float(MNIST_IMAGE_PIXELS))),
        name='weights')
    biases = tf.Variable(tf.zeros([hiddenlayer_units]),
                         name='biases')
    hidden1 = tf.nn.elu(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.name_scope('hiddenlayer2'):
    weights = tf.Variable(
        tf.truncated_normal([hiddenlayer_units, hiddenlayer_units],
                            stddev=1.0 / math.sqrt(float(hiddenlayer_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hiddenlayer_units]),
                         name='biases')
    hidden2 = tf.nn.elu(tf.matmul(hidden1, weights) + biases)
    
  with tf.name_scope('hiddenlayer3'):
    weights = tf.Variable(
        tf.truncated_normal([hiddenlayer_units, hiddenlayer_units],
                            stddev=1.0 / math.sqrt(float(hiddenlayer_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hiddenlayer_units]),
                         name='biases')
    hidden3 = tf.nn.elu(tf.matmul(hidden2, weights) + biases)
    
  with tf.name_scope('hiddenlayer4'):
    weights = tf.Variable(
        tf.truncated_normal([hiddenlayer_units, hiddenlayer_units],
                            stddev=1.0 / math.sqrt(float(hiddenlayer_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hiddenlayer_units]),
                         name='biases')
    hidden4 = tf.nn.elu(tf.matmul(hidden3, weights) + biases)
  
  with tf.name_scope('hiddenlayer5'):
    weights = tf.Variable(
        tf.truncated_normal([hiddenlayer_units, hiddenlayer_units],
                            stddev=1.0 / math.sqrt(float(hiddenlayer_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hiddenlayer_units]),
                         name='biases')
    hidden5 = tf.nn.elu(tf.matmul(hidden4, weights) + biases)
    
  with tf.name_scope('hiddenlayer6'):
    weights = tf.Variable(
        tf.truncated_normal([hiddenlayer_units, hiddenlayer_units],
                            stddev=1.0 / math.sqrt(float(hiddenlayer_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hiddenlayer_units]),
                         name='biases')
    hidden6 = tf.nn.elu(tf.matmul(hidden5, weights) + biases)

  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([hiddenlayer_units, 10])
    b_fc2 = bias_variable([10])

    y_output = tf.matmul(hidden6, W_fc2) + b_fc2
  return y_output


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1/math.sqrt(float(hiddenlayer_units)))
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def appstart(stri):
  # Import data
  mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  if(stri=="relu"):
      y_output  = deepnnwithrelu(x)
  elif(stri=="elu"):
      y_output  = deepnnwithelu(x)
  else:
      y_output  = deepnnwithsigmoid(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                            logits=y_output)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_output, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  graph_location = tempfile.mkdtemp()
  print('Saving graph to: %s' % graph_location)
  train_writer = tf.summary.FileWriter(graph_location)
  train_writer.add_graph(tf.get_default_graph())
  resultarray =[]
  iterarray=[]
  accarray=[]

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(40000):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1]})
        print('step %d, training accuracy %g' % (i, train_accuracy))
        iterarray.append(i)
        accarray.append(train_accuracy)
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    resultarray.append(iterarray)
    resultarray.append(accarray)
    
    testaccuracy = accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels})    
    print('test accuracy %g' %testaccuracy)
    resultarray.append(testaccuracy)
  return resultarray  

rarray =[]
rarray.append(appstart("sigmoid"))
rarray.append(appstart("relu"))
rarray.append(appstart("elu"))
fig = plt.figure()
axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(rarray[0][0],rarray[0][1],'r')
axes.plot(rarray[0][0],rarray[1][1],'b')
axes.plot(rarray[0][0],rarray[2][1],'g')

axes.set_xlabel('Train Iterations')
axes.set_ylabel('Train accuracy')
plt.plot()