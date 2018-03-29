import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
from scipy.misc import imresize
import time

N = 1024
i=-1
X = list()
# read pm2_5 values
with open('data_01_2018.csv', 'r') as f:
    next(f)
    for row in f:
        res = row.split(',')
        timerange, map_idx, pm2_5 = int(res[0]), int(res[1]), float(res[2])
        # create list of pm2_5
        if (timerange != i):
            x1 = list()
            for j in range(N):
                x1.append(0)
            i = timerange
            X.append(x1)
        else:
            x1 = X[timerange]
        x1[map_idx] = pm2_5

# read AQI label
y = list()
with open('aqi_01_2018', 'r') as f:
    for row in f:
        y.append(int(row))

# show values
m = timerange + 1
X = np.asarray(X)
print(X.shape)
y = np.asarray(y)
y = y[0:m]
print(y.shape)
y_ = np.copy(y)
y_[np.argwhere(y <= 15)[:,0]] = 0
y_[np.argwhere(np.logical_and(y > 15, y <= 35))[:,0]] = 1
y_[np.argwhere(np.logical_and(y > 35, y <= 75))[:,0]] = 2
y_[np.argwhere(y > 75)[:,0]] = 3
#print(np.count_nonzero(y_ == 0))
#print(np.count_nonzero(y_ == 1))
#print(np.nonzero(y_ == 2))
#print(np.count_nonzero(y_ == 3))

#Visualizing CIFAR 10
X = X.reshape(m, 32, 32)
X0 = X[4]
points = list()
values = list()
for i in range(32):
    for j in range(32):
        if X0[i,j] > 0:
            points.append([i,j])
            values.append(X0[i,j])
points = np.asarray(points)
values = np.asarray(values).reshape(-1,1)
print(points.shape)
print(values.shape)

from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=3, weights='distance')
neigh.fit(points, values)
print(neigh.predict([28,25])[0][0])

'''norm = c.Normalize()
for i in range(40, 69):
    fig = plt.figure()
    #i = np.random.choice(range(len(X)))
    ax = fig.add_subplot(121)
    ax.set_title('i = ' + str(i) + ', y = ' + str(y_[i]))
    X_i = X[i:i+1][0]
    #X_i = imresize(X_i, (227, 227))
    ax.imshow(X_i, cmap='gray', norm=norm, interpolation='bilinear')

    #i = np.random.choice(range(len(X)))
    ax = fig.add_subplot(122)
    ax.set_title('i = ' + str(i+1) + ', y = ' + str(y_[i+1]))
    X_i1 = X[i+1:i+2][0]
    #X_i1 = imresize(X_i1, (227, 227))
    ax.imshow(X_i1, cmap='gray', norm=norm, interpolation='bilinear')

    plt.show()'''


