#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 02:13:08 2017

@author: aditya
"""

from tensorflow.examples.tutorials.mnist import input_data
import argparse
import os
import sys
import time
from six.moves import xrange
import tensorflow as tf
import math
MNIST_IMAGE_SIZE = 28
MNIST_IMAGE_PIXELS =28*28
OUTPUT_CLASSES = 10 
Batch_Size = 100
LEARNING_RATE = 0.01
def modelusingrelu(images):
  """Build the MNIST model up to where it may be used for inference.
  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.
  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  hiddenlayer_units =32
  # Hidden 1
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
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hiddenlayer_units, OUTPUT_CLASSES],
                            stddev=1.0 / math.sqrt(float(hiddenlayer_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([OUTPUT_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden6, weights) + biases
  return logits


def Xentropyloss(logits, labels):
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')

def training(loss, learning_rate):
  tf.summary.scalar('loss', loss)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  correct = tf.nn.in_top_k(logits, labels, 1)
  return tf.reduce_sum(tf.cast(correct, tf.int32))


print(tf.__version__)
mnist = input_data.read_data_sets('MNIST_data',False)
sess = tf.InteractiveSession()
with tf.Graph().as_default():
    images = tf.placeholder(tf.float32, shape=(Batch_Size, MNIST_IMAGE_PIXELS))
    labels = tf.placeholder(tf.float32, shape=(Batch_Size))
    logits = modelusingrelu(images)
    loss = Xentropyloss(logits, labels)
    training = training(loss,LEARNING_RATE)
    evaluation = evaluation(logits, labels)
    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter("logs", sess.graph)
    sess.run(init)
    for step in xrange(2000):
      start_time = time.time()
      images_feed, labels_feed = mnist.next_batch(Batch_Size)
      feed_dict = {
                      images: images_feed,
                      labels: labels_feed,
                  }

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([training, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == 2000:
        checkpoint_file = os.path.join("logs", 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        evaluation(sess,
                evaluation,
                images,
                labels,
                mnist.train)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        evaluation(sess,
                evaluation,
                images,
                labels,
                mnist.validation)
        # Evaluate against the test set.
        print('Test Data Eval:')
        evaluation(sess,
                evaluation,
                images,
                labels,
                mnist.test)
    