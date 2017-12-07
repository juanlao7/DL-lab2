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
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM, GRU
from keras.optimizers import RMSprop, SGD
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import argparse
import json
import time
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os
import pickle
import sys
from keras.utils import plot_model

__author__ = 'bejar, lao'

FUTURE = 6

def plotOptions(results, title, ylabel, keys):
    plt.gca().set_color_cycle(None)
    plt.plot([], '--', color='black')
    plt.plot([], color='black')
    
    for i in results:
        plt.plot(results[i]['h'][keys[0]])
    
    plt.legend(['Training', 'Validation'] + results.keys(), loc='upper right')
    
    plt.gca().set_color_cycle(None)
    
    for i in results:
        plt.plot(results[i]['h'][keys[1]], '--')
    
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Epoch')
    
    plt.ylim(ymin=0.02, ymax=0.06)
    
    plt.show()
    
# Makes the dataset compatible with keras stateful LSTM's, where the ith element of the batch uses the state stored for the ith element of the previous batch.
def batchify((x, y), batchSize):
    x, y = removeSurplus((x, y), batchSize)
    x = batchifyImpl(x, batchSize)
    y = batchifyImpl(y, batchSize)
    return x, y

def batchifyImpl(data, batchSize):
    n = len(data)
    parsedData = np.empty(data.shape)
    k = 0
    
    for i in xrange(batchSize):
        for j in xrange(i, n, batchSize):
            parsedData[k] = data[j]
            k += 1
    
    return parsedData

def removeSurplus((x, y), batchSize):
    surplus = x.shape[0] % batchSize
    
    if surplus == 0:
        return x, y
    
    return x[:-surplus], y[:-surplus]

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
    return np.stack(lvect, axis=1)

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

def reshapeY(y):
    return np.reshape(y, (y.shape[0], 1, y.shape[1] * y.shape[2]))

if __name__ == '__main__':
    start = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config', help='Experiment configuration')
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true', default=False)
    parser.add_argument('--gpu', help="Use LSTM/GRU gru implementation", action='store_true', default=False)
    args = parser.parse_args()

    verbose = 1 if args.verbose else 0
    impl = 2 if args.gpu else 0
    
    vars = {0: 'wind_speed', 1: 'air_density', 2: 'temperature', 3: 'pressure'}

    wind = np.load('Wind.npz')
    wind = wind['90-45142']

    scaler = StandardScaler()
    wind = scaler.fit_transform(wind)

    configs = load_config_file(args.config)

    results = {}
    
    for configName in configs:
        print('###    ' + configName + '    ###')
        
        resultsDir = 'results/' + args.config
        
        try:
            os.makedirs(resultsDir)
        except:
            pass
        
        resultFileName = resultsDir + '/' + configName + '.result'
        
        if os.path.isfile(resultFileName):
            handler = open(resultFileName, 'rb')
            results[configName] = pickle.load(handler)
            handler.close()
            continue
        
        config = configs[configName]
    
        # Size of the training and size for validatio+test set (half for validation, half for test)
        datasize = config['datasize']
        testsize = config['testsize']
    
        # Length of the lag for the training window
        lag = config['lag']
    
        wind_train = wind[:datasize, :]
        train = lagged_vector(wind_train, lag=lag + FUTURE - 1)
        train_x, train_y = train[:, :-FUTURE], train[:, -FUTURE:]
        train_y = reshapeY(train_y)
        
        wind_test = wind[datasize:datasize + testsize, :]
        test = lagged_vector(wind_test, lag=lag + FUTURE - 1)
        half_test = int(test.shape[0] / 2)
    
        val_x, val_y = test[:half_test, :-FUTURE], test[:half_test, -FUTURE:]
        val_y = reshapeY(val_y)
        
        test_x, test_y = test[half_test:, :-FUTURE], test[half_test:, -FUTURE:]
        test_y = reshapeY(test_y)
        
        batch_size = config['batch']
        sequential = ('sequential' in config and config['sequential'])
        ySize = train_y.shape[2]
        
        if sequential:
            if lag != 1:
                raise Exception('When using the sequential structure, lag must be 1!')
            
            train_x, train_y = batchify((train_x, train_y), batch_size)
            val_x, val_y = batchify((val_x, val_y), batch_size)
            test_x, test_y = batchify((test_x, test_y), batch_size)
        else:
            train_y = np.reshape(train_y, (train_y.shape[0], train_y.shape[2]))
            val_y = np.reshape(val_y, (val_y.shape[0], val_y.shape[2]))
            test_y = np.reshape(test_y, (test_y.shape[0], test_y.shape[2]))
    
        ############################################
        # Model
    
        neurons = config['neurons']
        drop = config['drop']
        interdrop = config['interdrop'] if 'interdrop' in config else 0
        nlayers = config['nlayers']
        RNN = LSTM if config['rnn'] == 'LSTM' else GRU
    
        activation = config['activation']
        activation_r = config['activation_r']
    
        model = Sequential()
        
        batchSizeForInput = batch_size if sequential else None
        
        if nlayers == 1:
            model.add(RNN(neurons, batch_input_shape=(batchSizeForInput, train_x.shape[1], train_x.shape[2]), implementation=impl, dropout=drop, activation=activation, recurrent_activation=activation_r, stateful=sequential, return_sequences=sequential))
        else:
            model.add(RNN(neurons, batch_input_shape=(batchSizeForInput, train_x.shape[1], train_x.shape[2]), implementation=impl, dropout=drop, activation=activation, recurrent_activation=activation_r, return_sequences=True, stateful=sequential))
            model.add(Dropout(interdrop))
            
            for i in range(1, nlayers-1):
                model.add(RNN(neurons, dropout=drop, implementation=impl, activation=activation, recurrent_activation=activation_r, return_sequences=True, stateful=sequential))
                model.add(Dropout(interdrop))
                
            model.add(RNN(neurons, dropout=drop, implementation=impl, activation=activation, recurrent_activation=activation_r, stateful=sequential, return_sequences=sequential))

        model.add(Dropout(interdrop))
        model.add(Dense(ySize))
        
            ############################################
        # Training
    
        learning_rate = config['learning_rate'] if 'learning_rate' in config else 0.0001
        clipping_norm = config['clipping_norm'] if 'clipping_norm' in config else None
        optimizer = RMSprop(lr=learning_rate, clipnorm=clipping_norm)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        
        plot_model(model, to_file='model.png', show_shapes=True)
    
        nepochs = config['epochs']
        validation_patience = config['validation_patience']
        
        stopper = EarlyStopping(monitor='val_loss', patience=validation_patience)
    
        h = model.fit(train_x, train_y,
                  batch_size=batch_size,
                  epochs=nepochs,
                  callbacks=[stopper],
                  verbose=verbose,
                  validation_data=(val_x, val_y),
                  shuffle=not sequential)
    
        ############################################
        # Results
    
        score = model.evaluate(test_x, test_y, batch_size=batch_size, verbose=0)
        
        results[configName] = {
            'h': h.history,
            'test_loss': score
        }
        
        handler = open(resultFileName, 'wb')
        pickle.dump(results[configName], handler)
        handler.close()
    
    print '### FINISH! ###'
    end = time.time()
    print 'Elapsed time:', (end - start) / 60, 'minutes'

    for i in results:
        h = results[i]['h']
        print i, '(' + str(len(h['loss'])), 'epochs):'
        result = [str(round(i, 6)) for i in [h['loss'][-1], h['val_loss'][-1], results[i]['test_loss']]]
        print ','.join(result)
        
    # Plotting
    plotOptions(results, 'Model loss', 'Loss', ['val_loss', 'loss'])

