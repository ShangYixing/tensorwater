


# -*- coding: utf-8 -*-
"""
Introducing TensorFlow into molecular simulation
"""
import functools
import numpy as np
import tensorflow as tf
# from tensorflow.python import debug as tf_debug


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.

    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class Model:
    '''definite a model to calculate prediction, optimize, loss, and error'''
    def __init__(self, image, label):
        '''a function to initial some args

        :argument:self.image; self.label; self.prediction; self.optimize; self.loss; self.error
        :param image:
        :param label:
        '''
        self.image = image
        self.label = label
        self.prediction
        self.optimize
        self.loss
        self.error

    @define_scope(initializer=tf.contrib.slim.xavier_initializer())
    def prediction(self):
        '''a function to calculate prediction

        :return: x
        '''
        x = self.image
        x = tf.contrib.slim.fully_connected(x, 10)
        x = tf.contrib.slim.fully_connected(x, 1, activation_fn=None)
        return x

    @define_scope
    def loss(self):
        '''a function to calculate loss and narrow the gap

        :return:loss
        '''
        loss = tf.reduce_sum(tf.square(self.prediction - self.label))
        return loss

    @define_scope
    def optimize(self):
        '''a function to calculate optimize by using optimizer

        :return:optimizer.minimize(self.loss)
        '''
        optimizer = tf.train.AdamOptimizer(1.)
        return optimizer.minimize(self.loss)

    @define_scope
    def error(self):
        '''a function to calculate mistakes

        :return: tf.reduce_mean(tf.cast(mistakes,tf.float32))
        '''
        mistakes = tf.not_equal(
            tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        return tf.reduce_mean(tf.cast(mistakes, tf.float32))