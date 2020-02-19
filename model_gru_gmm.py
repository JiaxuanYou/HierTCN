import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from args import *
from utils import *
from customized_tcn_cell import *
from customed_gru_cell import *


# gru+gmm
def model_gru_gmm(args, x, x_gap=None, x_impression=None, name='rnn_gmm', reuse=tf.AUTO_REUSE):
    """
    given a batch of sequence x, pred a batch of gaussian mixture model with n gaussian.
    for each batch and time point, the pred has length n * (1 + d + d), where '1' stands for
    the size of class prior (pi), the 2nd 'd' stands for size of the mean (mu), and the 3rd 'd' stands for
    the size of variance (sigma).
    :param x: (b,t,d)
    :param args: object that contains all setup parameters (import from args.py)
    :param x_gap: (b,t,1)
    :param x_impression: (b,t,k)
    :param name: default
    :param reuse: default
    :return: pred: if learn sigma: the shape is (b,t,n*(1+d*2)) else: the shape is (b,t,n*(1+d))
    """
    with tf.variable_scope(name, reuse=reuse):
        # input
        if args.has_gap and x_gap is not None:
            x_weight = tf.exp(-x_gap / args.gap_bandwidth)
            x = tf.concat([x, x_weight], axis=2)
        if args.has_impression and x_impression is not None:
            x = tf.concat([x, tf.reduce_mean(x_impression, axis=2)], axis=2)

        rnn_layers = []
        for i in range(args.num_layer):
            if args.has_batchnorm:
                rnn_layers.append(GRUCellBN(args.hidden_dim, has_weightnorm=args.has_weightnorm))
                multi_rnn_cell = MultiRNNCellBN(rnn_layers)
            else:
                rnn_layers.append(rnn.GRUCell(args.hidden_dim))
                multi_rnn_cell = rnn.MultiRNNCell(rnn_layers)

        pred, state = tf.nn.dynamic_rnn(
            multi_rnn_cell,
            x,
            dtype=tf.float32,
            sequence_length=length(x)
        )
        # gmm
        if not args.sigma: # learn sigma
            n_out = args.num_mixture * (1 + args.output_dim + args.output_dim) # gmm param dim
        else:
            n_out = args.num_mixture * (1 + args.output_dim)  # gmm param dim
        pred = tf.contrib.layers.fully_connected(pred, num_outputs=n_out, activation_fn=None)
        return pred


def length(sequence):
    """
    compute actual length of sequence (non-zero timesteps) in each batch
    :param sequence: (b,t,d)
    :return: length: (b,)
    """
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2)) # b*t
    length = tf.reduce_sum(used, 1)# b
    length = tf.cast(length, tf.int32)
    return length