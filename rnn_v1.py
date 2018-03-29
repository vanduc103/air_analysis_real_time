from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import matplotlib.pyplot as plt
import random

# random number
seed = 128
rng = np.random.RandomState(seed)

def batch_creator(batch_size, dataset_length, timesteps):
    batch_x = list()
    batch_y = list()

    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice( dataset_length - timesteps - pred_timesteps, batch_size )
    for i in range(len(batch_mask)):
        offset = batch_mask[i]
        
        batch_x.append(X_train[ offset : offset + timesteps ])
        batch_y.append(X_train[ offset + timesteps + (pred_timesteps-1) ])
    
    return np.asarray(batch_x), np.asarray(batch_y)

def read_data(aqi_file):
    X = list()
    # read AQI data
    with open(aqi_file, 'r') as f:
        for row in f:
            X.append(int(row))

    # convert to numpy array
    X = np.asarray(X)
    return X

def RNN(x, weights, biases):

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # 1-layer LSTM with n_hidden units.
    rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

# training data
X09 = read_data('aqi_09_2017')
X10 = read_data('aqi_10_2017')
X11 = read_data('aqi_11_2017')
X12 = read_data('aqi_12_2017')
X_train = np.concatenate([X09, X10, X11, X12], axis = 0)

# split to train set and validate set
split_size = int(X_train.shape[0]*0.7)
X_train, X_val = X_train[:split_size], X_train[split_size:]
print(X_train.shape)
print(X_val.shape)

# testing data
X_test = read_data('aqi_01_2018')
print(X_test.shape)

# Training Parameters
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 1000

model_path = "model/rnn_aqi_model_v1.ckpt"
output_path = "model/output1.txt"
outfile = open(output_path, 'a')

# Network Parameters
num_input = 1 #
timesteps = 1 # timesteps
pred_timesteps = 12 # predict timesteps
n_hidden = 128 # hidden layer num of features

outfile.write('\n')
outfile.write("Result with timesteps = " + str(timesteps) + ", n_hidden = " + str(n_hidden) + "\n")

# tf Graph input
x = tf.placeholder("float", [None, timesteps, num_input])
y = tf.placeholder("float", [None, 1])

# RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, 1]))
}
biases = {
    'out': tf.Variable(tf.random_normal([1]))
}

# Prepare for training
pred = RNN(x, weights, biases)

# Loss and optimizer
cost = tf.losses.mean_squared_error(labels=y, predictions=pred)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# Start training
'''with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        # Make the training batch for each step
        batch_x, batch_y = batch_creator(batch_size, X_train.shape[0], timesteps)
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        batch_y = batch_y.reshape((batch_size, 1))

        # Run optimization
        _, loss, next_y = sess.run([optimizer, cost, pred], feed_dict={x: batch_x, y: batch_y})

        # Print result
        if step % display_step == 0 or step == 1:
            print("Iter = " + str(step) + ", RMSE = " + \
                  "{:.6f}".format(loss))
                
    print("Optimization Finished!")
    # Save model weights to disk
    save_path = saver.save(sess, model_path)
    print("Model saved in file: %s" % save_path)

    # Validate error
    loss_val = 0
    val_steps = 0
    for offset in range( X_val.shape[0] - timesteps - pred_timesteps ):
        val_steps += 1
        val_x = X_val[ offset : offset + timesteps ].reshape((1, timesteps, num_input))
        val_y = X_val[ offset + timesteps + (pred_timesteps-1) ].reshape((1, 1))

        loss, _ = sess.run([cost, pred], feed_dict={x: val_x, y: val_y})
        loss_val += loss

    # Print validate error
    print("Validate RMSE - {:.6f}".format(loss_val/val_steps))
    outfile.write("Validate RMSE - {:.6f}\n".format(loss_val/val_steps))'''

# Running a new session
print("---------------------------")
print("Starting testing session...")
with tf.Session() as sess:
    # Initialize variables
    sess.run(init)

    # Restore model weights from previously saved model
    saver.restore(sess, model_path)
    print("Model restored!")

    l_pred_y = list()
    l_actual_y = list()
    offset = 0
    print("Pred_y   Actual_y")
    #outfile.write("Pred_y   Actual_y\n")
    '''for i in range( X_test.shape[0] - timesteps - pred_timesteps ):
        test_x = X_test[ i + offset : i + offset + timesteps].reshape((1, timesteps, num_input))
        actual_y = X_test[ i + offset + timesteps + (pred_timesteps-1) ]
        pred_y = sess.run(pred, feed_dict={x: test_x})[0][0]

        print("{:.2f}".format(pred_y) + "      " + str(actual_y))
        #outfile.write("{:.2f}".format(pred_y) + "      " + str(actual_y) + "\n")

        l_pred_y.append(pred_y)
        l_actual_y.append(actual_y)'''
    test_x_base = X_test[offset:offset+timesteps]
    for i in range(pred_timesteps):
        test_x = test_x_base[i:i+timesteps].reshape((1, timesteps, num_input))
        actual_y = X_test[offset+i+timesteps]
        pred_y = sess.run(pred, feed_dict={x: test_x})[0][0]

        print("{:.2f}".format(pred_y) + "      " + str(actual_y))

        test_x_base = np.append(test_x_base, pred_y)
        l_pred_y.append(pred_y)
        l_actual_y.append(actual_y)

    # Plot
    plt.plot(l_actual_y, color='black')
    plt.plot(l_pred_y, color='red')
    plt.show()

