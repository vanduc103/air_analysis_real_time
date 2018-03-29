import os

for n_hidden in [64, 128, 256, 512]:
    for timesteps in [12, 24, 48, 96]:
        for s in range(10, -1, -1):
            print('python rnn_v3.py ' + str(timesteps) + ' ' + str(n_hidden) + ' ' + str(float(s)/10))
            os.system('python rnn_v3.py ' + str(timesteps) + ' ' + str(n_hidden) + ' ' + str(float(s)/10))

