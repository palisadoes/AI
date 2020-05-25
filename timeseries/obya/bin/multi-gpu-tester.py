#!/usr/bin/env python3
"""LSTM for sinusoidal data problem with regression framing.

Based on:

https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

"""

# Standard imports
import argparse
import math
from collections import namedtuple
import os
import sys

# PIP3 imports
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

tf.debugging.set_log_device_placement(True)


def setup():
    """Setup TensorFlow 2 operating parameters.

    Args:
        None

    Returns:
        result: Processor namedtuple of GPUs and CPUs in the system

    """
    # Initialize key variables
    memory_limit = 1024
    Processors = namedtuple('Processors', 'gpus, cpus')
    gpu_names = []
    cpu_names = []

    # Reduce error logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Limit Tensorflow v2 Limit GPU Memory usage
    # https://www.tensorflow.org/guide/gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    cpus = tf.config.experimental.list_physical_devices('CPU')
    if bool(gpus) is True:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for _, gpu in enumerate(gpus):
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=memory_limit)])
                gpu_names.append(gpu.name.replace('physical_device', ''))

            # Currently, memory growth needs to be the same across GPUs
            for _, cpu in enumerate(cpus):
                cpu_names.append(cpu.name.replace('physical_device', ''))

        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Return
    result = Processors(gpus=gpu_names, cpus=cpu_names)
    return result


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


def main():

    # Setup memory
    processors = setup()

    # fix random seed for reproducibility
    numpy.random.seed(7)

    # Get CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gpus',
        help='Number of GPUs to use.',
        type=int, default=1)
    args = parser.parse_args()
    gpus = args.gpus

    # load the dataset
    dataframe = DataFrame(
        [0.00000, 5.99000, 11.92016, 17.73121, 23.36510, 28.76553, 33.87855,
         38.65306, 43.04137, 46.99961, 50.48826, 53.47244, 55.92235, 57.81349,
         59.12698, 59.84970, 59.97442, 59.49989, 58.43086, 56.77801, 54.55785,
         51.79256, 48.50978, 44.74231, 40.52779, 35.90833, 30.93008, 25.64279,
         20.09929, 14.35496, 8.46720, 2.49484, -3.50245, -9.46474, -15.33247,
         -21.04699, -26.55123, -31.79017, -36.71147, -41.26597, -45.40815,
         -49.09663, -52.29455, -54.96996, -57.09612, -58.65181, -59.62146,
         -59.99540, -59.76988, -58.94716, -57.53546, -55.54888, -53.00728,
         -49.93605, -46.36587, -42.33242, -37.87600, -33.04113, -27.87613,
         -22.43260, -16.76493, -10.92975, -4.98536, 1.00883, 6.99295, 12.90720,
         18.69248, 24.29100, 29.64680, 34.70639, 39.41920, 43.73814, 47.62007,
         51.02620, 53.92249, 56.28000, 58.07518, 59.29009, 59.91260, 59.93648,
         59.36149, 58.19339, 56.44383, 54.13031, 51.27593, 47.90923, 44.06383,
         39.77815, 35.09503, 30.06125, 24.72711, 19.14590, 13.37339, 7.46727,
         1.48653, -4.50907, -10.45961, -16.30564, -21.98875, -27.45215,
         -32.64127, -37.50424, -41.99248, -46.06115, -49.66959, -52.78175,
         -55.36653, -57.39810, -58.85617, -59.72618, -59.99941, -59.67316,
         -58.75066, -57.24115, -55.15971, -52.52713, -49.36972, -45.71902,
         -41.61151, -37.08823, -32.19438, -26.97885, -21.49376, -15.79391,
         -9.93625, -3.97931, 2.01738, 7.99392, 13.89059, 19.64847, 25.21002,
         30.51969, 35.52441, 40.17419, 44.42255, 48.22707, 51.54971, 54.35728,
         56.62174, 58.32045, 59.43644, 59.95856, 59.88160, 59.20632, 57.93947,
         56.09370, 53.68747, 50.74481, 47.29512, 43.37288, 39.01727, 34.27181,
         29.18392, 23.80443, 18.18710, 12.38805, 6.46522, 0.47779, -5.51441,
         -11.45151])
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    # reshape input to be [samples, time steps, features]
    trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    # Create and fit the LSTM network
    if abs(gpus) > len(processors.gpus):
        devices = processors.gpus
    else:
        devices = processors.gpus[:min(0, len(processors.gpus) - 1)]

    strategy = tf.distribute.MirroredStrategy(devices=devices)
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Define the model
    with strategy.scope():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(4, input_shape=(1, look_back)))
        model.add(tf.keras.layers.Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')

    model.fit(
        trainX,
        trainY,
        epochs=100,
        batch_size=int(dataset.size * len(devices) / 20), 
        verbose=2)

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))

    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[
        len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset), label='Complete Data')
    plt.plot(trainPredictPlot, label='Training Data')
    plt.plot(testPredictPlot, label='Prediction Data')
    plt.legend(loc='upper left')
    plt.title('Using {} GPUs'.format(gpus))
    plt.show()


if __name__ == "__main__":
    main()
