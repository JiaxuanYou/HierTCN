import tensorflow as tf
from args import *
from utils import *
from customized_tcn_cell import *
from customed_gru_cell import *

# Notations
# b: batch size, eg, 32
# t: length of timesteps, eg, 100,1000
# d: pin dimension, eg, 512
# k: number of impressions, eg 20

def model_tcn(args, x, x_gap=None, x_impression=None, name='tcn',reuse=tf.AUTO_REUSE,training=tf.constant(True),mask=None):
    """
    given a batch of sequence x, pred a batch of sequence y_hat using TCN (https://arxiv.org/pdf/1803.01271.pdf)
    :param x: (b,t,d)
    :param args: object that contains all setup parameters (import from args.py)
    :param x_gap: (b,t,1)
    :param x_impression: (b,t,k)
    :param name: default
    :param reuse: default
    :param training: whether the mode is training or testing
    :param mask: (b,1): whether a user changes in the batch
    :return: pred: (b,t,d)
    """
    with tf.variable_scope(name, reuse=reuse):
        tcn = TemporalConvNet(num_channels=args.tcn_channel, kernel_size=args.kernel_size, dropout=args.dropout)

        if args.has_gap and x_gap is not None:
            x_weight = tf.exp(-x_gap / args.gap_bandwidth)
            x = tf.concat([x, x_weight],axis=2)
        if args.has_impression and x_impression is not None:
            x = tf.concat([x, tf.reduce_mean(x_impression,axis=2)],axis=2)

        x = dense(x, units=128, activation=None, use_bias=False, has_weightnorm=args.has_weightnorm, reuse=tf.AUTO_REUSE,name='emb')
        pred = tcn(x,training=training,mask=mask)
        # pred = tf.contrib.layers.fully_connected(pred,num_outputs=args.output_dim,activation_fn=None)
        if args.model_low_type=='tcn_multi':
            pred = dense(pred,units=args.output_dim*args.multi_num,activation=None,has_weightnorm=args.has_weightnorm)
        else:
            pred = dense(pred,units=args.output_dim,activation=None,has_weightnorm=args.has_weightnorm)
        if args.l2_normalize:
            pred = tf.nn.l2_normalize(pred,dim=-1)
        return pred