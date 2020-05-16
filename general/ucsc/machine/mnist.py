"""Class that processes MNIST data."""

import os
import struct
from array import array


class MNIST(object):
    """Class that processes MNIST data."""

    def __init__(self, path='.'):
        """Initialize the class.

        Args:
            path: Path to unzipped MNIST data

        Returns:
            None

        """
        # Load data
        self.path = path

        self.test_img_fname = 't10k-images-idx3-ubyte'
        self.test_lbl_fname = 't10k-labels-idx1-ubyte'

        self.train_img_fname = 'train-images-idx3-ubyte'
        self.train_lbl_fname = 'train-labels-idx1-ubyte'

        self.test_images = []
        self.test_labels = []

        self.train_images = []
        self.train_labels = []

    def load_testing(self):
        """Return processed testing data.

        Args:
            None:

        Returns:
            (ims, labels): Tuple containing a list of image and label data.

        """
        # Load data
        ims, labels = self.load(os.path.join(self.path, self.test_img_fname),
                                os.path.join(self.path, self.test_lbl_fname))

        self.test_images = ims
        self.test_labels = labels

        return ims, labels

    def load_training(self):
        """Return processed training data.

        Args:
            None:

        Returns:
            (ims, labels): Tuple containing a list of image and label data.

        """
        # Load data
        ims, labels = self.load(os.path.join(self.path, self.train_img_fname),
                                os.path.join(self.path, self.train_lbl_fname))

        self.train_images = ims
        self.train_labels = labels

        return ims, labels

    @classmethod
    def load(cls, path_img, path_lbl):
        """Load data from image and label files.

        Args:
            path_img: Path to image file
            path_lbl: Path to label file

        Returns:
            (ims, labels): Tuple containing a list of image and label data.

        """
        # Load data
        with open(path_lbl, 'rb') as f_hdl:
            magic, size = struct.unpack(">II", f_hdl.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", f_hdl.read())

        with open(path_img, 'rb') as f_hdl:
            magic, size, rows, cols = struct.unpack(">IIII", f_hdl.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", f_hdl.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels

    @classmethod
    def display(cls, img, width=28, threshold=200):
        """Render image files for display.

        Args:
            None:

        Returns:
            (ims, labels): Tuple containing a list of image and label data.

        """
        # Load data
        render = ''
        for i in range(len(img)):
            if i % width == 0:
                render += '\n'
            if img[i] > threshold:
                render += '@'
            else:
                render += '.'
        return render
