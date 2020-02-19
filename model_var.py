import tensorflow as tf
from args import *
from utils import *

# Notations
# b: batch size, eg, 32
# t: length of timesteps, eg, 100,1000
# d: pin dimension, eg, 512
# k: number of impressions, eg 20

def model_var(args, x, name='var', reuse=tf.AUTO_REUSE):
    """
    vector auto regressive model
    :param x: (b,t,d)
    :param args: object that contains all setup parameters (import from args.py)
    :return: pred : (b,t,d)
    """
    with tf.variable_scope(name, reuse=reuse):
        # vector w and bias b
        w = tf.Variable(tf.zeros([1, args.lag]), name="weight") #[1, lag]
        w = tf.expand_dims(w, 0)
        w = tf.expand_dims(w, 0) # [1, 1, 1, lag]
        w = tf.tile(w, [tf.shape(x)[0], tf.shape(x)[1], 1, 1]) # [batch, time, 1, lag]
        b = tf.Variable(tf.zeros([1, args.input_dim], name='bias'))
        b = tf.expand_dims(b,0) # [1,1,dim]
        b = tf.tile(b, [tf.shape(x)[0], tf.shape(x)[1], 1]) # [batch,time,dim]
        pred = tf.matmul(w, x) # [batch, time, 1, dim]
        pred = tf.squeeze(pred, axis=2) + b # [batch, time, dim]
        return pred