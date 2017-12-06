#!/usr/bin/env python3
"""Script to demonstrate  basic tensorflow machine learning."""

import tensorflow as tf


def main():
    """Main Function.

    Display data prediction from tensorflow model

    """
    # Initialize key variables
    x1 = tf.constant(5)
    x2 = tf.constant(6)

    # Stage the tensors to run
    result = tf.multiply(x1, x2)
    print(result)

    # Defines our session and launches graph
    with tf.Session() as sess:
        # Get result
        output = sess.run(result)
        print(output)


if __name__ == "__main__":
    main()
