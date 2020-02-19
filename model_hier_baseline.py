import tensorflow as tf
from tensorflow.contrib import rnn
from args import *
from utils import *
from customized_tcn_cell import *
from customed_gru_cell import *

from model_gru import *
from model_gru_gmm import *
from model_tcn import *
from model_tcn_gmm import *
from model_mv import *


# Notations
# b: batch size, eg, 32
# t: length of timesteps, eg, 100,1000
# d: pin dimension, eg, 512
# D: topic dimension, eg, 50
# k: number of impressions, eg 20
# m: number of sessions, eg 10
# s: length of session, eg 100


def model_hier_baseline(args,x,y,mask,state,x_gap=None,x_impression=None,name='hier',reuse=tf.AUTO_REUSE,training=tf.constant(True)):
    """
    A gru hier baseline in https://arxiv.org/pdf/1706.04148.pdf
    the lower level model is GRU and it passes the output of GRU to the high level
    :param x: m*(b,s,d)
    :param y: m*(b,s,d)
    :param mask: m*(b,1)
    :param state: (b,num_hidden_dim*num_layer)
    :return: pred,state
    """
    with tf.variable_scope(name,reuse=reuse):
        # init high level rnn
        rnn_layers = []
        for i in range(args.num_layer):
            if args.has_batchnorm:
                rnn_layers.append(GRUCellBN(args.hidden_dim,has_weightnorm=args.has_weightnorm))
                multi_rnn_cell = MultiRNNCellBN(rnn_layers, state_is_tuple=False)
            else:
                rnn_layers.append(rnn.GRUCell(args.hidden_dim))
                multi_rnn_cell = rnn.MultiRNNCell(rnn_layers,state_is_tuple=False)
        # loop over user sessions
        for i in range(len(x)):
            if args.has_gap:
                # apply decay
                state *= tf.exp(-x_gap[i] / args.gap_bandwidth)
            mask_y = tf.sign(tf.reduce_sum(tf.abs(y[i]), 2))  # B*T
            # low level model input
            # x_slice = x[i]
            x_slice = dense(x[i], units=128, use_bias=False, activation=None, has_weightnorm=args.has_weightnorm,reuse=tf.AUTO_REUSE,name='emb')
            if args.data_noise is not None:
                x_slice += tf.random_normal(shape=tf.shape(x_slice), mean=0.0, stddev=args.data_noise, dtype=tf.float32)
            if args.has_impression:
                impression_slice = tf.reduce_mean(x_impression[i],axis=2)
                x_slice = tf.concat((x_slice, impression_slice), axis=2)
            # low level model
            if 'tcn' in args.model_low_type:
                pred = model_tcn(args,x_slice,training=training,mask=mask_y)
            elif args.model_low_type == 'gru':
                state_low = dense(state,units=args.hidden_dim * args.num_layer,activation=None,has_weightnorm=args.has_weightnorm,reuse=tf.AUTO_REUSE)
                pred,state_low = model_gru(args,x_slice,initial_state=state_low,return_state=True)
            elif args.model_low_type == 'mv':
                # x_pad = tf.contrib.layers.fully_connected(state,num_outputs=args.output_dim,activation_fn=None)
                x_pad = dense(state,units=args.output_dim,activation=None,has_weightnorm=args.has_weightnorm)
                pred = model_mv_tf(x[i],n=args.mv_n,x_pad=x_pad)
            # save pred for each session
            if i==0:
                pred_all = pred
            else:
                pred_all = tf.concat((pred_all,pred),axis=1)
            if args.has_batchnorm:
                output, state = multi_rnn_cell(state_low, state, training=training)
            else:
                output, state = multi_rnn_cell(state_low, state)
            # apply mask
            state *= mask[i]
        return pred_all,state

