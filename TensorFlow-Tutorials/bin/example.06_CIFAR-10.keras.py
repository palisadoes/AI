#!/usr/bin/env python3
'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os


class KerasCNN(object):
    """Support vector machine class."""

    def __init__(self):
        """Instantiate the class.

        Args:
            train_batch_size: Training batch size

        Returns:
            None

        """
        # Initialize variables
        batch_size = 32
        num_classes = 10
        epochs = 100
        fill = 30
        data_augmentation = True
        save_dir = '/tmp/saved_models'
        model_name = 'keras_cifar10_trained_self.model.h5'

        filter_size1 = 32
        filter_size2 = 64
        convolution_window = (3, 3)
        fully_connected_layer_features = 512

        # The data, split between train and test sets:
        (x_train, y_train), (self.x_test, self.y_test) = cifar10.load_data()
        size_train = x_train.shape[0]
        size_test = self.x_test.shape[0]
        input_shape = x_train.shape[1:]
        print('{0: <{1}} {2}'.format('x_train shape:', fill, x_train.shape))
        print('{0: <{1}} {2}'.format('Train samples:', fill, size_train))
        print('{0: <{1}} {2}'.format('Test samples:', fill, size_test))
        print(
            '{0: <{1}} {2} {3}x{4} pixels, {5} colors'.format(
                'Input shape:', fill,
                input_shape, input_shape[0], input_shape[1], input_shape[2]))

        # Get the number of steps per epoch
        steps_per_epoch = int(size_train / epochs)

        # Convert class vectors to binary class matrices.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        self.y_test = keras.utils.to_categorical(self.y_test, num_classes)

        self.model = Sequential()
        self.model.add(
            Conv2D(filter_size1,
                   convolution_window,
                   padding='same', input_shape=x_train.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(filter_size1, convolution_window))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(
            Conv2D(filter_size2, convolution_window, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(filter_size2, convolution_window))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))

        self.model.add(Flatten())
        self.model.add(Dense(fully_connected_layer_features))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(num_classes))
        self.model.add(Activation('softmax'))

        # initiate RMSprop optimizer
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        # Let's train the model using RMSprop
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])

        x_train = x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        x_train /= 255
        self.x_test /= 255

        if not data_augmentation:
            print('Not using data augmentation.')
            self.model.fit(
                x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(self.x_test, self.y_test),
                shuffle=True)
        else:
            print('Using real-time data augmentation.')
            # This will do preprocessing and realtime data augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                zca_epsilon=1e-06,  # epsilon for ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                # randomly shift images horizontally (fraction of total width)
                width_shift_range=0.1,
                # randomly shift images vertically (fraction of total height)
                height_shift_range=0.1,
                shear_range=0.,  # set range for random shear
                zoom_range=0.,  # set range for random zoom
                channel_shift_range=0.,  # set range for random channel shifts
                # set mode for filling points outside the input boundaries
                fill_mode='nearest',
                cval=0.,  # value used for fill_mode = "constant"
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False,  # randomly flip images
                # set rescaling factor (applied before any other transformation)
                rescale=None,
                # set function that will be applied on each input
                preprocessing_function=None,
                # image data format, either "channels_first" or "channels_last"
                data_format=None,
                # fraction of images reserved for validation (strictly between 0 and 1)
                validation_split=0.0)

            # Compute quantities required for feature-wise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(x_train)

            # Fit the model on the batches generated by datagen.flow().
            self.model.fit_generator(
                datagen.flow(x_train, y_train, batch_size=batch_size),
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=(self.x_test, self.y_test),
                workers=4)

        # Save model and weights
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, model_name)
        self.model.save(model_path)
        print('Saved trained model at {}'.format(model_path))


def main():
    """Run main function."""
    # Score trained model.
    cnn = KerasCNN()
    scores = cnn.model.evaluate(cnn.x_test, cnn.y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


if __name__ == "__main__":
    main()
