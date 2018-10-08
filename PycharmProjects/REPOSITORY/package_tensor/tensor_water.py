# -*- coding: utf-8 -*-
"""
Introducing TensorFlow into molecular simulation
"""


import numpy as np
import tensorflow as tf
import Model as Md

# from tensorflow.python import debug as tf_debug




def main():
    '''this is a founction of training prediction

    read two data files
    according to a [800,6] matrix
    training 5000000 times
    bring predicted value closer to the true values
    :keyword: images, labels, outloss
    :return: outloss
    '''
    image = tf.placeholder(tf.float32, [None, 6])
    label = tf.placeholder(tf.float32, [None, 1])
    model = Md.Model(image, label)           #changed
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #sess.run(image)
    images = np.genfromtxt('oneline_structures_train.csv', delimiter=',')
    images = np.reshape(images, (800, 6))
    labels = np.genfromtxt('oneline_energies_train.csv', delimiter=',')
    labels = np.reshape(labels, (800, 1))
    for _ in range(5000000):  # train 5000000 times
        sess.run(model.optimize, {image: images, label: labels})
        outloss = sess.run(model.loss, {image: images, label: labels})
        print(outloss)



if __name__ == '__main__':
    main()
