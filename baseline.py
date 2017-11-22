"""
.. module:: WindPrediction

WindPrediction
*************

:Description: WindPrediction

:Authors: bejar, lao
    

:Version: 

:Created on: 06/09/2017 9:47

:Updated on: 22/11/2017 0:20 

"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop, SGD
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import argparse
import json
import time
from keras.callbacks import EarlyStopping

__author__ = 'bejar'

def lagged_vector(data, lag=1):
    """
    Returns a matrix with columns that are the steps of the lagged time series
    Last columns are the value to predict
    :param data:
    :param lag:
    :return:
    """
    lvect = []
    for i in xrange(lag):
        lvect.append(data[i: -lag+i])
    lvect.append(data[lag:])
    lvect = np.stack(lvect, axis=1)
    return np.reshape(lvect, (lvect.shape[0], lvect.shape[1] * lvect.shape[2]))

def load_config_file(nfile, abspath=False):
    """
    Read the configuration from a json file

    :param abspath:
    :param nfile:
    :return:
    """
    ext = '.json' if 'json' not in nfile else ''
    pre = '' if abspath else './'
    fp = open(pre + nfile + ext, 'r')

    s = ''

    for l in fp:
        s += l

    return json.loads(s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config', help='Experiment configuration')
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true', default=False)
    parser.add_argument('--gpu', help="Use LSTM/GRU gru implementation", action='store_true', default=False)
    args = parser.parse_args()

    verbose = 1 if args.verbose else 0
    impl = 2 if args.gpu else 0

    config = load_config_file(args.config)

    print("Starting:", time.ctime())

    ###########################################
    # Data

    vars = {0: 'wind_speed', 1: 'air_density', 2: 'temperature', 3: 'pressure'}
    nvars = len(vars)

    wind = np.load('Wind.npz')
    print(wind.files)
    wind = wind['90-45142']

    scaler = StandardScaler()
    wind = scaler.fit_transform(wind)

    # Size of the training and size for validatio+test set (half for validation, half for test)
    datasize = config['datasize']
    testsize = config['testsize']

    # Length of the lag for the training window
    lag = config['lag']

    wind_train = wind[:datasize, :]
    train = lagged_vector(wind_train, lag=lag)
    train_x, train_y = train[:, :-nvars], train[:, -nvars:]
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))

    wind_test = wind[datasize:datasize+testsize, :]
    test = lagged_vector(wind_test, lag=lag)
    half_test = int(test.shape[0] / 2)

    val_x, val_y = test[:half_test, :-nvars], test[:half_test, -nvars:]
    val_x = np.reshape(val_x, (val_x.shape[0], val_x.shape[1], 1))

    test_x, test_y = test[half_test:, :-nvars], test[half_test:, -nvars:]
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    ############################################
    # Model

    neurons = config['neurons']
    drop = config['drop']
    nlayers = config['nlayers']
    RNN = LSTM if config['rnn'] == 'LSTM' else GRU

    activation = config['activation']
    activation_r = config['activation_r']

    model = Sequential()
    if nlayers == 1:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], 1), implementation=impl, dropout=drop,
                      activation=activation, recurrent_activation=activation_r))
    else:
        model.add(RNN(neurons, input_shape=(train_x.shape[1], 1), implementation=impl, dropout=drop,
                      activation=activation, recurrent_activation=activation_r, return_sequences=True))
        for i in range(1, nlayers-1):
            model.add(RNN(neurons, dropout=drop, implementation=impl,
                          activation=activation, recurrent_activation=activation_r, return_sequences=True))
        model.add(RNN(neurons, dropout=drop, implementation=impl,
                      activation=activation, recurrent_activation=activation_r))
    model.add(Dense(nvars))

    print('lag: ', lag, 'Neurons: ', neurons, 'Layers: ', nlayers, activation, activation_r)
    print()

    ############################################
    # Training

    optimizer = RMSprop(lr=0.0001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    batch_size = config['batch']
    nepochs = config['epochs']
    validation_patience = config['validation_patience']
    
    stopper = EarlyStopping(monitor='val_loss', patience=validation_patience)

    model.fit(train_x, train_y,
              batch_size=batch_size,
              epochs=nepochs,
              callbacks=[stopper],
              verbose=verbose, validation_data=(val_x, val_y))

    ############################################
    # Results

    print()
    score = model.evaluate(val_x, val_y,
                           batch_size=batch_size,
                           verbose=0)
    print('MSE Val= ', score)
    print ('MSE Val persistence =', mean_squared_error(val_y[1:], val_y[0:-1]))

    score = model.evaluate(test_x, test_y,
                           batch_size=batch_size,
                           verbose=0)
    print('MSE Test= ', score)
    print ('MSE Test persistence =', mean_squared_error(test_y[1:], test_y[0:-1]))
    print()
    print("Ending:", time.ctime())
