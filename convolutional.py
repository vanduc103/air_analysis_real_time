import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_dense = labels_dense.astype(int)
    labels_one_hot[np.arange(num_labels), labels_dense] = 1
    
    return labels_one_hot

def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    #temp_batch = unclean_batch_x / unclean_batch_x.max()
    temp_batch = unclean_batch_x / 150.0
    #temp_batch = unclean_batch_x
    
    return temp_batch

def batch_creator(batch_size, dataset_length, num_classes=10):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = train_x[[batch_mask]]
    batch_x = preproc(batch_x)

    batch_y = train_y[[batch_mask]]    
    batch_y = dense_to_one_hot(batch_y, num_classes)
    
    return batch_x, batch_y

def read_data(data_file, aqi_file):
    N = 1024 # number of cells
    timerange_init = -1
    X = list()
    # read pm2_5 values
    with open(data_file, 'r') as f:
        next(f) # by pass the header
        for row in f:
            res = row.split(',')
            timerange, map_idx, pm2_5 = int(res[0]), int(res[1]), float(res[2])
            # create list of pm2_5
            if (timerange != timerange_init):
                x1 = list()
                for j in range(N):
                    x1.append(0)
                timerange_init = timerange
                X.append(x1)
            else:
                x1 = X[timerange]
            x1[map_idx] = pm2_5

    # read AQI label
    Y = list()
    with open(aqi_file, 'r') as f:
        for row in f:
            Y.append(float(row))

    # show values
    m = timerange + 1
    X = np.asarray(X)
    X = X.reshape(m, 32, 32, 1)
    Y = np.asarray(Y)
    Y = Y[0:m]
    y_ = np.copy(Y)
    y_[np.argwhere(Y <= 15)[:,0]] = 0
    y_[np.argwhere(np.logical_and(Y > 15, Y <= 35))[:,0]] = 1
    y_[np.argwhere(np.logical_and(Y > 35, Y <= 75))[:,0]] = 2
    y_[np.argwhere(Y > 75)[:,0]] = 3

    return X, y_

# random number
seed = 128
rng = np.random.RandomState(seed)

# read train data
X2, y2 = read_data('data/data_12_2017.csv', 'data/aqi_12_2017')
X3, y3 = read_data('data/data_11_2017.csv', 'data/aqi_11_2017')
X4, y4 = read_data('data/data_10_2017.csv', 'data/aqi_10_2017')
X5, y5 = read_data('data/data_09_2017.csv', 'data/aqi_09_2017')
X_train = np.concatenate([X5, X4, X3, X2], axis = 0)
y_train = np.concatenate([y5, y4, y3, y2], axis = 0)
print(X_train.shape)
print(y_train.shape)
print(np.count_nonzero(y_train == 0))
print(np.count_nonzero(y_train == 1))
print(np.count_nonzero(y_train == 2))
print(np.count_nonzero(y_train == 3))

# split to train set and validate set
split_size = int(X_train.shape[0]*0.9)
#np.random.shuffle(X_train)
#y_train = y_train[X_train[:,0]]

train_x, val_x = X_train[:split_size], X_train[split_size:]
train_y, val_y = y_train[:split_size], y_train[split_size:]
print('Read training data done !')

# read test data
X_test, y_test = read_data('data/data_01_2018.csv', 'data/aqi_01_2018')
print(X_test.shape)
print(y_test.shape)
print('Read testing data done !')

### define the layers
image_size = 32
filter_num = 32
filter_size = 3
fc3_size = 384
fc4_size = 192
output_classes = 4

# define placeholders
x = tf.placeholder(tf.float32, [None, image_size, image_size, 1])
y = tf.placeholder(tf.float32, [None, output_classes])

# set remaining variables
epochs = 100
batch_size = 32
learning_rate = 0.0001

### weight initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


### define model
# convolution-pooling layer define
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# convolution-pooling layer #1
W_conv1 = weight_variable([filter_size, filter_size, 1, filter_num])
b_conv1 = bias_variable([filter_num])
conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
pool1 = max_pool_2x2(conv1)

# convolution-pooling layer #2
W_conv2 = weight_variable([filter_size, filter_size, 32, filter_num])
b_conv2 = bias_variable([filter_num])
conv2 = tf.nn.relu(conv2d(pool1, W_conv2) + b_conv2)
pool2 = max_pool_2x2(conv2)

# fully connected #3
pool2_reshape = tf.reshape(pool2, [-1, 8*8*32])
pool2_dim = pool2_reshape.get_shape()[1].value
W_fc3 = tf.get_variable(name='W_fc3', shape=[pool2_dim, fc3_size], 
		initializer=tf.contrib.layers.xavier_initializer())
b_fc3 = tf.Variable(tf.zeros(fc3_size))
fc3 = tf.nn.relu(tf.add(tf.matmul(pool2_reshape, W_fc3), b_fc3))

# fully connected #4
W_fc4 = tf.get_variable(name='W_fc4', shape=[fc3_size, fc4_size],   
                initializer=tf.contrib.layers.xavier_initializer())
b_fc4 = tf.Variable(tf.zeros(fc4_size))
fc4 = tf.nn.relu(tf.add(tf.matmul(fc3, W_fc4), b_fc4))

# output layer
W_output = tf.get_variable(name='W_output', shape=[fc4_size, output_classes],   
                initializer=tf.contrib.layers.xavier_initializer())
b_output = tf.Variable(tf.zeros(output_classes))
output_layer = tf.add(tf.matmul(fc4, W_output), b_output)

### loss function - cross entropy with softmax
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))

## optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# initialize all variables
init = tf.initialize_all_variables()

# create session and run neural network on session
print('\n Training start ...')
with tf.Session() as sess:
  sess.run(init)
  
  ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize
  graph_x = list()
  graph_y = list()
  for epoch in range(epochs):
    avg_cost = 0
    total_batch = int(train_x.shape[0]/batch_size)
    l_conv1, l_conv2 = list(), list()
    for i in range(total_batch):
      batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], output_classes)
      _, c = sess.run([optimizer, loss], feed_dict = {x: batch_x, y: batch_y})
            
      avg_cost += c / total_batch
    
    print("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))
    # compute accuracy on validate set
    prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
    print("Validation accuracy:", accuracy.eval({x: preproc(val_x), y: dense_to_one_hot(val_y, output_classes)}))
    graph_x.append(epoch+1)
    graph_y.append(avg_cost)

  print("\nTraining complete!")

  # show graph of loss value
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Graph of loss value')
  plt.grid(True)
  plt.plot(graph_x, graph_y)
  plt.show()

  # compute accuracy on test set
  print('\nTesting ...\n')
  prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
  print("Test Accuracy:", accuracy.eval({x: preproc(X_test), y: dense_to_one_hot(y_test, output_classes)}))
  
  # view some testing images
  for i in range(2):
    batch_mask = rng.choice(len(X_test), 1)
    batch_x = X_test[[batch_mask]]
    batch_x = preproc(batch_x)
    d_ouput_layer = sess.run([output_layer], feed_dict = {x: batch_x})
    pred = np.argmax(output_layer, 0)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.set_title('#' + str(i) + ': prediction = ' + str(pred))
    ax.imshow(batch_x[0][:,:,0], cmap='gray', interpolation='bilinear')

    batch_mask = rng.choice(len(X_test), 1)
    batch_x = X_test[[batch_mask]]
    batch_x = preproc(batch_x)
    d_ouput_layer = sess.run([output_layer], feed_dict = {x: batch_x})
    pred = np.argmax(output_layer, 0)

    ax = fig.add_subplot(122)
    ax.set_title('#' + str(i + 1) + ': prediction = ' + str(pred))
    ax.imshow(batch_x[0][:,:,0], cmap='gray', interpolation='bilinear')
    plt.show()

