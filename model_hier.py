import tensorflow as tf
from tensorflow.contrib import rnn
from args import *
from utils import *
from customized_tcn_cell import *
from customed_gru_cell import *
from model_mv import *
from model_tcn import *
from model_tcn_gmm import *
from model_gru import *
from model_gru_gmm import *


# Notations
# b: batch size, eg, 32
# t: length of timesteps, eg, 100,1000
# d: pin dimension, eg, 512
# k: number of impressions, eg 20


def model_hier(args,x,y,mask,state,x_gap=None,x_impression=None,name='hier',reuse=tf.AUTO_REUSE,training=tf.constant(True)):
    """
    :param x: k*(B,S,d), batch*session_len*512
    :param y: k*(B,S,d), batch*session_len*512
    :param state: state = B*(hidden*layer), batch*512
    :return: pred : (b,d), state: (b,hidden*layer)
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
                if args.train_gap:
                    gap_mu = dense(state, units=1, activation=tf.nn.sigmoid,
                          has_weightnorm=args.has_weightnorm, reuse=tf.AUTO_REUSE, name='gap_mu')
                    state *= tf.exp(-x_gap[i] / (args.gap_bandwidth*gap_mu))
                else:
                    state *= tf.exp(-x_gap[i] / args.gap_bandwidth)
            mask_y = tf.sign(tf.reduce_sum(tf.abs(y[i]), 2))  # B*T
            # low level model input
            x_slice = dense(x[i], units=128, activation=None, use_bias=False, has_weightnorm=args.has_weightnorm,reuse=tf.AUTO_REUSE,name='emb')
            # print(x_slice.get_shape())
            if args.data_noise is not None:
                x_slice += tf.random_normal(shape=tf.shape(x_slice), mean=0.0, stddev=args.data_noise, dtype=tf.float32)
            feature_slice = tf.tile(tf.expand_dims(state,axis=1),[1,tf.shape(x_slice)[1],1])
            x_slice = tf.concat((x_slice,feature_slice),axis=-1)
            # x_slice = dense(x_slice, units=256, activation=None,
            #                 has_weightnorm=args.has_weightnorm,reuse=tf.AUTO_REUSE,name='x_slice')
            if args.has_impression:
                impression_slice = tf.reduce_mean(x_impression[i],axis=2)
                x_slice = tf.concat((x_slice, impression_slice), axis=2)
            # low level model
            if args.model_low_type == 'tcn':
                pred = model_tcn(args,x_slice,training=training,mask=mask_y)
            elif args.model_low_type == 'tcn_gmm':
                pred = model_tcn_gmm(args,x_slice,training=training,mask=mask_y)
            elif args.model_low_type == 'gru':
                pred = model_gru(args,x_slice)
            elif args.model_low_type == 'gru_gmm':
                pred = model_gru_gmm(args,x_slice)
            elif args.model_low_type == 'mv':
                # x_pad = tf.contrib.layers.fully_connected(state,num_outputs=args.output_dim,activation_fn=None)
                x_pad = dense(state,units=args.output_dim,activation=None,
                              has_weightnorm=args.has_weightnorm,reuse=tf.AUTO_REUSE,name='x_pad')
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
            y_slice = dense(y_slice, units=128, activation=None, has_weightnorm=args.has_weightnorm,reuse=tf.AUTO_REUSE,name='emb')
            # print(y_slice.get_shape())
            # high level model update
            if args.has_batchnorm:
                output, state = multi_rnn_cell(y_slice, state, training=training)
            else:
                output, state = multi_rnn_cell(y_slice, state)
            # apply mask
            state *= mask[i]
        return pred_all,state

