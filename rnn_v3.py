from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt
import random
import time
import sys

seed = 128
rng = np.random.RandomState(seed)
random.seed(seed)

def batch_creator(batch_size, dataset_length, timesteps):
    batch_x = list()
    batch_y = list()
    batch_x_w = list()

    """Create batch with random samples and return appropriate format"""
    #batch_mask = rng.choice(dataset_length-timesteps, batch_size)
    offset_rand = random.randrange(0, dataset_length - timesteps - batch_size - pred_timesteps)
    for i in range(batch_size):
        offset = offset_rand + i
        batch_x.append(X_train[offset:offset+timesteps])
        batch_y.append(X_train[offset+timesteps])
        x_w = np.concatenate([temp_train[offset+timesteps], rain_train[offset+timesteps], wind_speed_train[offset+timesteps], wind_direction_train[offset+timesteps]], axis = 0)
        # make time, date onehot
        time_onehot = time2onehot(offset+timesteps)
        date_onehot = date2onehot_train(offset+timesteps)
        x_w = np.concatenate([x_w, np.concatenate([time_onehot, date_onehot], axis=0)], axis=0)
        
        batch_x_w.append(x_w)
    
    return np.asarray(batch_x), np.asarray(batch_x_w), np.asarray(batch_y)

def read_data(aqi_file):
    X = list()
    # read AQI data
    with open(aqi_file, 'r') as f:
        for row in f:
            X.append(int(row))

    # convert to numpy array
    X = np.asarray(X)
    return X

def read_weather(weather_file):
    data = list()
    with open(weather_file, 'r') as f:
        content = f.read().splitlines()
    content = content[1:] # skip header
    content = [content[i].split(',') for i in range(len(content))]
    content = np.array(content)
    temp = content[:,0].astype(float)
    rain = content[:,1].astype(float)
    wind_speed = content[:,2].astype(float)
    wind_direction = content[:,4]
    
    return (temp, rain, wind_speed, wind_direction)

def time2onehot(timeidx):
    t = timeidx - (timeidx/24)*24
    result = np.zeros(24)
    result[t] = 1
    return result

def date2onehot_train(timeidx):
    basetime = time.mktime((2017, 9, 1, 0, 0, 0, 0, 1, 1)) # 2017-09-01 00:00:00
    d = time.localtime(basetime + timeidx * 3600)
    wday = d.tm_wday

    result = np.zeros(2)
    holidays = [276, 277, 278, 279, 282, 359]
    # weekend and holiday
    if wday == 5 or wday == 6 or d.tm_yday in holidays:
        result[1] = 1
    else:
        result[0] = 1
    return result

def date2onehot_test(timeidx):
    basetime = time.mktime((2018, 1, 1, 0, 0, 0, 0, 1, 1)) # 2018-01-01 00:00:00
    d = time.localtime(basetime + timeidx * 3600)
    wday = d.tm_wday

    result = np.zeros(2)
    holidays = [1]
    # weekend and holiday
    if wday == 5 or wday == 6 or d.tm_yday in holidays:
        result[1] = 1
    else:
        result[0] = 1
    return result

# Training data
X09 = read_data('data/aqi_09_2017')
X10 = read_data('data/aqi_10_2017')
X11 = read_data('data/aqi_11_2017')
X12 = read_data('data/aqi_12_2017')
X_train = np.concatenate([X09, X10, X11, X12], axis = 0)

# Split to train set and validate set
split_size = int(X_train.shape[0]*0.7)
X_train, X_val = X_train[:split_size], X_train[split_size:]
print(X_train.shape)
print(X_val.shape)

# Testing data
X_test = read_data('data/aqi_01_2018')
print(X_test.shape)

# Weather data
w09 = read_weather('data/weather_09_2017.csv')
w10 = read_weather('data/weather_10_2017.csv')
w11 = read_weather('data/weather_11_2017.csv')
w12 = read_weather('data/weather_12_2017.csv')
w_test = read_weather('data/weather_01_2018.csv')

temp = np.concatenate([w09[0], w10[0], w11[0], w12[0]], axis = 0).reshape(-1, 1)
rain = np.concatenate([w09[1], w10[1], w11[1], w12[1]], axis = 0).reshape(-1, 1)
wind_speed = np.concatenate([w09[2], w10[2], w11[2], w12[2]], axis = 0).reshape(-1, 1)
wind_direction = np.concatenate([w09[3], w10[3], w11[3], w12[3]], axis = 0)

# Scaler
scaler_temp = StandardScaler().fit(temp)
scaler_rain = StandardScaler().fit(rain)
scaler_wind_speed = StandardScaler().fit(wind_speed)

temp = scaler_temp.transform(temp)
rain = scaler_rain.transform(rain)
wind_speed = scaler_wind_speed.transform(wind_speed)

# Convert to OneHotEncoding
le = LabelEncoder().fit(wind_direction.ravel())
enc = OneHotEncoder(sparse=False)
wind_direction = enc.fit_transform(le.transform(wind_direction).reshape(-1,1))
print(wind_direction.shape)

# Split to training and validating data
temp_train, temp_val = temp[:split_size], temp[split_size:]
rain_train, rain_val = rain[:split_size], rain[split_size:]
wind_speed_train, wind_speed_val = wind_speed[:split_size], wind_speed[split_size:]
wind_direction_train, wind_direction_val = wind_direction[:split_size], wind_direction[split_size:]

# Weather testing data
temp_test = w_test[0][0:X_test.shape[0]].reshape(-1, 1)
rain_test = w_test[1][0:X_test.shape[0]].reshape(-1, 1)
wind_speed_test = w_test[2][0:X_test.shape[0]].reshape(-1, 1)
wind_direction_test = w_test[3][0:X_test.shape[0]]

# Transform testing data
temp_test = scaler_temp.transform(temp_test)
rain_test = scaler_rain.transform(rain_test)
wind_speed_test = scaler_wind_speed.transform(wind_speed_test)
wind_direction_test = enc.transform(le.transform(wind_direction_test).reshape(-1,1))

# Training Parameters
learning_rate = 0.0001
training_steps = 20000
batch_size = 256
display_step = 1000
model_path = "model/rnn_aqi_model_v3.ckpt"
output_path = "model/output_v3.txt"
outfile = open(output_path, 'a')

# Network Parameters
num_input = 1 # rnn input dimension
timesteps = int(sys.argv[1]) #[24, 48, 96, 128] # timesteps
pred_timesteps = 12 # predict timesteps
n_hidden = int(sys.argv[2]) #[128, 256, 512] # hidden layer for rnn layer
w_input = 20+26 # other conditions input dimension
n_hidden_fc = 128 # hidden layer for fc layer
alpha = float(sys.argv[3])
outfile.write('\n')
outfile.write("Result with timesteps = " + str(timesteps) + ", n_hidden = " + str(n_hidden) + ", alpha = " + str(alpha) + "\n")

# tf Graph input
x = tf.placeholder("float", [None, timesteps, num_input])
x_w = tf.placeholder("float", [None, w_input])
y = tf.placeholder("float", [None, 1])

# Weights and biases
weights = {
    'rnn': tf.Variable(tf.random_normal([n_hidden, 1])),
    'fc': tf.Variable(tf.random_normal([w_input, n_hidden_fc])),
    'out': tf.Variable(tf.random_normal([n_hidden_fc, 1]))
}
biases = {
    'rnn': tf.Variable(tf.random_normal([1])),
    'fc': tf.Variable(tf.random_normal([n_hidden_fc])),
    'out': tf.Variable(tf.random_normal([1]))
}

def model(x, x_w, weights, biases):

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # 1-layer LSTM with n_hidden units.
    rnn_cell = rnn.BasicLSTMCell(n_hidden)

    # generate prediction
    outputs, states = rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # there are n_input outputs but
    # we only want the last output
    y_rnn = tf.matmul(outputs[-1], weights['rnn']) + biases['rnn']

    # weather information
    y_w_hidden = tf.matmul(x_w, weights['fc']) + biases['fc']
    y_w = tf.matmul(y_w_hidden, weights['out']) + biases['out']

    # add weather information to rnn output
    return tf.add(y_rnn*alpha, y_w*(1.0-alpha))

# Prepare for training
pred = model(x, x_w, weights, biases)

# Loss and optimizer
cost = tf.losses.mean_squared_error(labels=y, predictions=pred)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        # Make the training batch for each step
        batch_x, batch_x_w, batch_y = batch_creator(batch_size, X_train.shape[0], timesteps)
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        batch_y = batch_y.reshape((batch_size, 1))
        batch_x_w = batch_x_w.reshape((batch_size, w_input))

        # Run optimization
        _, loss, next_y = sess.run([optimizer, cost, pred], feed_dict={x: batch_x, x_w: batch_x_w, y: batch_y})

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
    for offset in range(X_val.shape[0] - timesteps - pred_timesteps):
        val_steps += 1
        val_X_base = X_val[offset:offset+timesteps]
        for i in range(pred_timesteps):
            val_x = val_X_base[i:i+timesteps].reshape((1, timesteps, num_input))
            val_y = X_val[offset+i+timesteps].reshape((1, 1))
            val_x_w = np.concatenate([temp_val[offset+i+timesteps], rain_val[offset+i+timesteps], wind_speed_val[offset+i+timesteps], wind_direction_val[offset+i+timesteps]], axis = 0)
            # make time, date onehot
            time_onehot = time2onehot(X_train.shape[0]+offset+i+timesteps)
            date_onehot = date2onehot_train(X_train.shape[0]+offset+i+timesteps)
            val_x_w = np.concatenate([val_x_w, np.concatenate([time_onehot, date_onehot], axis=0)], axis=0)
            val_x_w = val_x_w.reshape((1, w_input))

            loss, pred_y = sess.run([cost, pred], feed_dict={x: val_x, x_w: val_x_w, y: val_y})
            pred_y = pred_y[0][0]
            val_X_base = np.append(val_X_base, pred_y)
            loss_val += loss

    # Print validate error
    print("Validate RMSE - {:.6f}".format(loss_val/val_steps))
    outfile.write("Validate RMSE - {:.6f}\n".format(loss_val/val_steps))

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
    test_x_base = X_test[offset:offset+timesteps]
    print("Pred_y   Actual_y")
    outfile.write("Pred_y   Actual_y\n")
    for i in range(pred_timesteps):
        test_x = test_x_base[i:i+timesteps].reshape((1, timesteps, num_input))
        test_x_w = np.concatenate([temp_test[offset+i+timesteps], rain_test[offset+i+timesteps], wind_speed_test[offset+i+timesteps], wind_direction_test[offset+i+timesteps]], axis = 0)
        # make time, date onehot
        time_onehot = time2onehot(offset+i+timesteps)
        date_onehot = date2onehot_test(offset+i+timesteps)
        test_x_w = np.concatenate([test_x_w, np.concatenate([time_onehot, date_onehot], axis=0)], axis=0)

        test_x_w = test_x_w.reshape((1, w_input))
        actual_y = X_test[offset+i+timesteps]

        pred_y = sess.run(pred, feed_dict={x: test_x, x_w: test_x_w})[0][0]
        print("{:.2f}".format(pred_y) + "      " + str(actual_y))
        outfile.write("{:.2f}".format(pred_y) + "      " + str(actual_y) + "\n")

        test_x_base = np.append(test_x_base, pred_y)
        l_pred_y.append(pred_y)
        l_actual_y.append(actual_y)

    # Plot
    plt.plot(l_actual_y, color='black')
    plt.plot(l_pred_y, color='red')
    #plt.show()

outfile.close()

