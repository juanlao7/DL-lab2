"""
.. module:: WindPrediction

WindPrediction
*************

:Description: WindPrediction

:Authors: bejar, lao
    

:Version: 

:Created on: 06/09/2017 9:47 

"""

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU, Dropout
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
    
    plt.ylim(ymin=0.025, ymax=0.037)
    
    plt.show()

def lagged_vector(data, lag=1):
    """
    Returns a matrix with columns that are the steps of the lagged time series
    Last column is the value to predict
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

if __name__ == '__main__':
    start = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config', help='Experiment configuration')
    parser.add_argument('--verbose', help="Verbose output (enables Keras verbose output)", action='store_true', default=False)
    parser.add_argument('--gpu', help="Use LSTM/GRU gru implementation", action='store_true', default=False)
    args = parser.parse_args()

    verbose = 1 if args.verbose else 0
    impl = 2 if args.gpu else 0

    ###########################################
    # Data

    vars = {0: 'wind_speed', 1: 'air_density', 2: 'temperature', 3: 'pressure'}

    wind = np.load('Wind.npz')
    wind = wind['90-45142'][:, 0:1]
    
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
        train = lagged_vector(wind_train, lag=lag)
        train_x, train_y = train[:, :-1], train[:, -1]
        train_y = train_y[:, 0]
        
        wind_test = wind[datasize:datasize + testsize, :]
        test = lagged_vector(wind_test, lag=lag)
        half_test = int(test.shape[0]/2)
    
        val_x, val_y = test[:half_test, :-1], test[:half_test,-1]
        val_y = val_y[:, 0]
    
        test_x, test_y = test[half_test:, :-1], test[half_test:,-1]
        test_y = test_y[:, 0]
        
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
        if nlayers == 1:
            model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl, dropout=drop,
                          activation=activation, recurrent_activation=activation_r))
        else:
            model.add(RNN(neurons, input_shape=(train_x.shape[1], train_x.shape[2]), implementation=impl, dropout=drop,
                          activation=activation, recurrent_activation=activation_r, return_sequences=True))
            model.add(Dropout(interdrop))
            
            for i in range(1, nlayers-1):
                model.add(RNN(neurons, dropout=drop, implementation=impl,
                              activation=activation, recurrent_activation=activation_r, return_sequences=True))
                model.add(Dropout(interdrop))
                
            model.add(RNN(neurons, dropout=drop, implementation=impl,
                          activation=activation, recurrent_activation=activation_r))
        
        model.add(Dropout(interdrop))
        model.add(Dense(1))
    
        ############################################
        # Training
    
        clipping_norm = config['clipping_norm'] if 'clipping_norm' in config else None
        optimizer = RMSprop(lr=0.0001, clipnorm=clipping_norm)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
    
        batch_size = config['batch']
        nepochs = config['epochs']
        validation_patience = config['validation_patience']
        
        stopper = EarlyStopping(monitor='val_loss', patience=validation_patience)
    
    
        h = model.fit(train_x, train_y,
                  batch_size=batch_size,
                  epochs=nepochs,
                  callbacks=[stopper],
                  verbose=verbose,
                  validation_data=(val_x, val_y))
    
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

