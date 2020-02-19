import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from args import *
from utils import *
from customized_tcn_cell import *
from customed_gru_cell import *



# tcn+gmm
def model_tcn_gmm(args,x, x_gap=None, x_impression=None, name='tcn_gmm', reuse=tf.AUTO_REUSE,training=tf.constant(True), mask=None):
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
    :param training: whether the mode is training or testing
    :param mask: (b,1): whether a user changes in the batch
    :return: pred: if learn sigma: the shape is (b,t,n*(1+d*2)) else: the shape is (b,t,n*(1+d))
    """
    with tf.variable_scope(name, reuse=reuse):
        tcn = TemporalConvNet(num_channels=args.tcn_channel, kernel_size=args.kernel_size, dropout=args.dropout)
        # tcn = TemporalConvNet(num_channels=[512]*8, kernel_size=5, dropout=1.0-1e-6)

        if args.has_gap and x_gap is not None:
            x_weight = tf.exp(-x_gap / args.gap_bandwidth)
            x = tf.concat([x, x_weight], axis=2)
        if args.has_impression and x_impression is not None:
            x = tf.concat([x, tf.reduce_mean(x_impression, axis=2)], axis=2)

        pred = tcn(x, training=training, mask=mask)
        ## gmm
        if not args.sigma: # learn sigma
            n_out = args.num_mixture * (1 + args.output_dim + args.output_dim)  # gmm param dim
        else:
            n_out = args.num_mixture * (1 + args.output_dim)  # gmm param dim
        pred = tf.contrib.layers.fully_connected(pred, num_outputs=n_out, activation_fn=None)
        return pred