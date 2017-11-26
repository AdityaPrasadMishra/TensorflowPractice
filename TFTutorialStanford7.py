""" Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from 
the number of fire in the city of Chicago
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd
#import utils
from tensorflow.python import debug as tf_debug

#import utils

DATA_FILE = 'data/fire_theft.xls'

# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
#print (data)
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
X = tf.placeholder(tf.float64, name='X')
Y = tf.placeholder(tf.float64, name='Y')

# Step 3: create weight and bias, initialized to 0
w = tf.Variable(0.0, name='weights', dtype='float64')
u = tf.Variable(0.0, name="weights_2", dtype='float64')
b = tf.Variable(0.0, name='bias', dtype='float64')

# Step 4: build model to predict Y
#Y_predicted = X * w + b 
Y_predicted = X*X*w + X*u + b

# Step 5: use the square error as the loss function
loss = tf.square(Y - Y_predicted, name='loss')
#loss = utils.huber_loss(Y, Y_predicted)

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    # Step 7: initialize the necessary variables, in this case, w and b
    sess.run(tf.global_variables_initializer()) 
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    
    writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)
    
    # Step 8: train the model
    for i in range(10): # train the model 100 epochs
        total_loss = 0
        for x, y in data:
            # Session runs train_op and fetch values of loss
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y:y}) 
            total_loss += l
            if(i==0):
                print(total_loss)
                print(l)
        print('Epoch {0}: {1}'.format(i, total_loss/n_samples))
        print(total_loss)
        print(n_samples)

    # close the writer when you're done using it
    writer.close() 
    
    # Step 9: output the values of w and b
    w,u, b = sess.run([w,u, b]) 

# plot the results
X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w + b, 'r', label='Predicted data')
plt.legend()
plt.show()