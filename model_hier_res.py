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
from utils import *


# Notations
# b: batch size, eg, 32
# t: length of timesteps, eg, 100,1000
# d: pin dimension, eg, 512
# D: topic dimension, eg, 50
# k: number of impressions, eg 20
# m: number of sessions, eg 10
# s: length of session, eg 100


def model_hier_res(args,x,y,mask,state,x_gap=None,x_impression=None,name='hier',reuse=tf.AUTO_REUSE,training=tf.constant(True)):
    """
    :param x: m*(b,s,d)
    :param y: m*(b,s,d)
    :param mask: m*(b,1)
    :param state: (b,num_hidden_dim*num_layer)
    :return: pred,state
    """
    with tf.variable_scope(name,reuse=reuse):
        # loop over user sessions
        for i in range(len(x)):
            if args.has_gap:
                # apply decay
                state *= tf.exp(-x_gap[i] / args.gap_bandwidth)
            mask_y = tf.sign(tf.reduce_sum(tf.abs(y[i]), 2))  # B*T
            # low level model input
            x_slice = x[i]
            if args.data_noise is not None:
                x_slice += tf.random_normal(shape=tf.shape(x_slice), mean=0.0, stddev=args.data_noise, dtype=tf.float32)
            feature_slice = tf.tile(tf.expand_dims(state,axis=1),[1,tf.shape(x_slice)[1],1])
            x_slice = tf.concat((x_slice,feature_slice),axis=-1)
            x_slice = dense(x_slice, units=256, activation=None,
                            has_weightnorm=args.has_weightnorm, reuse=tf.AUTO_REUSE, name='x_slice')
            if args.has_impression:
                impression_slice = tf.reduce_mean(x_impression[i],axis=2)
                x_slice = tf.concat((x_slice, impression_slice), axis=2)
            # low level model
            if 'tcn' in args.model_low_type:
                pred = model_tcn(x_slice,training=training,mask=mask_y)
            elif args.model_low_type == 'gru':
                pred = model_gru(x_slice)
            elif args.model_low_type == 'mv':
                # x_pad = tf.contrib.layers.fully_connected(state,num_outputs=args.output_dim,activation_fn=None)
                x_pad = dense(state,units=args.output_dim,activation=None,has_weightnorm=args.has_weightnorm,reuse=tf.AUTO_REUSE)
                pred = model_mv_tf(x[i],n=args.lag,x_pad=x_pad)
            # save pred for each session
            if i==0:
                pred_all = pred
            else:
                pred_all = tf.concat((pred_all,pred),axis=1)
            # high level model input
            # y_slice = tf.reduce_max(y[i], axis=1)  # B*512
            # mean pool
            y_activity_count = tf.reduce_sum(tf.sign(tf.reduce_sum(tf.abs(y[i]), 2)),1,keep_dims=True)
            y_slice = tf.reduce_sum(y[i], axis=1)/y_activity_count  # B*512

            # high level model
            state_feature = dense(tf.concat((state, y_slice), axis=-1),units=args.hidden_dim*args.num_layer//2,
                                activation=tf.nn.relu,has_weightnorm=args.has_weightnorm,reuse=tf.AUTO_REUSE,name='feature1')
            state_delta = dense(state_feature, units=args.hidden_dim * args.num_layer,
                                activation=None, has_weightnorm=args.has_weightnorm,reuse=tf.AUTO_REUSE,name='feature2')
            delta_ratio = dense(state_feature, units=1,
                                activation=tf.nn.sigmoid, has_weightnorm=args.has_weightnorm,reuse=tf.AUTO_REUSE,name='feature3')
            state = state_delta * delta_ratio + state * (1-delta_ratio)
            # apply mask
            state *= mask[i]

        return pred_all,state

