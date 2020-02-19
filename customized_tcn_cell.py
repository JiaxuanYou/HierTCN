import tensorflow as tf
import numpy as np
from args import *
from customized_convolution_layer import *
from customized_dense_layer import *
# Redefining CausalConv1D to simplify its return values
class CausalConv1D(Conv1D):
    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 has_weightnorm = args.has_weightnorm,
                 **kwargs):
        super(CausalConv1D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='valid',
            data_format='channels_last',
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            trainable=trainable,
            has_weightnorm=has_weightnorm,
            name=name, **kwargs
        )

    def call(self, inputs):
        padding = (self.kernel_size[0] - 1) * self.dilation_rate[0]
        inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
        return super(CausalConv1D, self).call(inputs)


# tf.reset_default_graph()
# with tf.Graph().as_default() as g:
#     # x = tf.random_normal((2, 10, 1))  # (batch_size, length, channel)
#     value = np.array([1,2,300,4,5])[np.newaxis,:,np.newaxis].astype(np.float32)
#     x = tf.constant(value)  # (batch_size, length, channel)
#     with tf.variable_scope("tcn"):
#         conv = CausalConv1D(1, 3, activation=None,dilation_rate=2)
#     output = conv(x)
#     init = tf.global_variables_initializer()
#
# with tf.Session(graph=g) as sess:
#     # Run the initializer
#     sess.run(init)
#     output_val = sess.run(output)
#     print(output_val.shape)
#     print(output_val[0,:,:])

class TemporalBlock(tf.layers.Layer):
    def __init__(self, n_outputs, kernel_size, strides, dilation_rate, dropout=0.2,
                 trainable=True, name=None, dtype=None,
                 activity_regularizer=None, **kwargs):
        super(TemporalBlock, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        self.dropout = dropout
        self.n_outputs = n_outputs
        self.conv1 = CausalConv1D(
            n_outputs, kernel_size, strides=strides,
            dilation_rate=dilation_rate, activation=tf.nn.relu,
            name="conv1")
        self.conv2 = CausalConv1D(
            n_outputs, kernel_size, strides=strides,
            dilation_rate=dilation_rate, activation=tf.nn.relu,
            name="conv2")
        # self.conv1 = CausalConv1D(
        #     n_outputs, kernel_size, strides=strides,
        #     dilation_rate=dilation_rate, activation=None,
        #     name="conv1")
        # self.conv2 = CausalConv1D(
        #     n_outputs, kernel_size, strides=strides,
        #     dilation_rate=dilation_rate, activation=None,
        #     name="conv2")
        self.down_sample = None

    def build(self, input_shape):
        channel_dim = 2
        self.dropout1 = tf.layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        self.dropout2 = tf.layers.Dropout(self.dropout, [tf.constant(1), tf.constant(1), tf.constant(self.n_outputs)])
        if input_shape[channel_dim] != self.n_outputs:
            # self.down_sample = tf.layers.Conv1D(
            #     self.n_outputs, kernel_size=1,
            #     activation=None, data_format="channels_last", padding="valid")
            self.down_sample = Dense(self.n_outputs, activation=None, has_weightnorm=args.has_weightnorm)
        self.built = True

    def call(self, inputs, training=True,mask=None):
        x = self.conv1(inputs)

        if args.has_batchnorm:
            if args.model_type == 'hier':
                x = batch_norm_3d(x, training=training, mask=mask, name='bn_x',size=args.max_activity_len)
            else:
                x = batch_norm_3d(x, training=training, mask=mask, name='bn_x', size=args.max_seq_len)
        if args.has_layernorm:
            x = tf.contrib.layers.layer_norm(x)
        x = self.dropout1(x, training=training)
        # x = self.conv2(x)
        # x = tf.contrib.layers.layer_norm(x)
        # x = self.dropout2(x, training=training)
        if self.down_sample is not None:
            inputs = self.down_sample(inputs)
            if args.has_batchnorm:
                inputs = batch_norm_3d(inputs, training=training, mask=mask, name='bn_input',size=args.max_activity_len)
        return tf.nn.relu(x + inputs)
        # return x



"""### Temporal convolutional networks"""


class TemporalConvNet(tf.layers.Layer):
    '''
    Receptive field: 1+2*(kernel-1)*(2^n-1)
    '''
    def __init__(self, num_channels, kernel_size=2,strides=1, dropout=0.2,
                 trainable=True, name=None, dtype=None,
                 activity_regularizer=None, **kwargs):
        super(TemporalConvNet, self).__init__(
            trainable=trainable, dtype=dtype,
            activity_regularizer=activity_regularizer,
            name=name, **kwargs
        )
        self.layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            out_channels = num_channels[i]
            self.layers.append(
                TemporalBlock(out_channels, kernel_size, strides=strides, dilation_rate=dilation_size,
                              dropout=dropout, name="tblock_{}".format(i))
            )

    def call(self, inputs, training=tf.constant(True),mask=None):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, training=training,mask=mask)
        return outputs

# tf.reset_default_graph()
# g = tf.Graph()
# with g.as_default():
#     tf.set_random_seed(111)
#     Xinput = tf.placeholder(tf.float32, shape=[None, 10, 1])
#     tcn = TemporalConvNet([1]*6, 8, 0.25)
#     output = tcn(Xinput, training=tf.constant(True))
#     init = tf.global_variables_initializer()
#
# with tf.Session(graph=g) as sess:
#     # Run the initializer
#     sess.run(init)
#     a = np.random.randn(1, 10, 1)
#     a[0,5,0]=1000
#     res = sess.run(output, {Xinput: a})
#     print(res.shape)
#     print(res[0, :, 0])
#     # print(res[1, :, 1])


# def batch_norm_3d(x, training, mask=None, name='bn', epsilon=1e-3, decay=0.999, size=None):
#     '''customized batch_norm with mask
#         default: x=B*T*D, mask=B*T
#     '''
#
#     with tf.variable_scope(name):
#         if size is None:
#             size = x.get_shape().as_list()[1]
#
#         scale = tf.get_variable('scale', [size,x.get_shape()[-1]], initializer=tf.constant_initializer(0.1))
#         offset = tf.get_variable('offset', [size,x.get_shape()[-1]])
#
#         pop_mean = tf.get_variable('pop_mean', [size,x.get_shape()[-1]], initializer=tf.zeros_initializer, trainable=False)
#         pop_var = tf.get_variable('pop_var', [size,x.get_shape()[-1]], initializer=tf.ones_initializer, trainable=False)
#         # batch_mean, batch_var = tf.nn.moments(x, [0])
#         mask = tf.pad(mask,paddings=tf.constant([(0, 0,), (0, 1)]) * (size-tf.shape(mask)[1]))
#         mask = tf.expand_dims(mask,axis=-1)
#         x *= mask
#         batch_mean = tf.reduce_sum(x,axis=0)/(tf.reduce_sum(mask,axis=0)+epsilon)
#         batch_var = tf.reduce_sum(tf.square(x-batch_mean)*mask,axis=0)/(tf.reduce_sum(mask,axis=0)+epsilon)
#         train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
#         train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
#
#         def batch_statistics():
#             with tf.control_dependencies([train_mean_op, train_var_op]):
#                 return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)
#
#         def population_statistics():
#             return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)
#
#         return tf.cond(training, batch_statistics, population_statistics)


def batch_norm_3d(x, training, mask=None, name='bn', epsilon=1e-3, decay=0.999, size=None):
    '''customized batch_norm with mask
        default: x=B*T*D, mask=B*T
    '''

    with tf.variable_scope(name):
        if size is None:
            size = x.get_shape().as_list()[1]

        size_batch = tf.shape(x)[1]
        scale = tf.get_variable('scale', [size,x.get_shape()[-1]], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset', [size,x.get_shape()[-1]])
        scale_batch = scale[:size_batch,:]
        offset_batch = scale[:size_batch,:]


        pop_mean = tf.get_variable('pop_mean', [size,x.get_shape()[-1]], initializer=tf.zeros_initializer, trainable=False)
        pop_var = tf.get_variable('pop_var', [size,x.get_shape()[-1]], initializer=tf.ones_initializer, trainable=False)
        pop_mean_batch = pop_mean[:size_batch,:]
        pop_var_batch = pop_var[:size_batch,:]
        # batch_mean, batch_var = tf.nn.moments(x, [0])
        # mask = tf.pad(mask,paddings=tf.constant([(0, 0,), (0, 1)]) * (size-tf.shape(mask)[1]))
        mask = tf.expand_dims(mask,axis=-1)
        x *= mask
        batch_mean = tf.reduce_sum(x,axis=0)/(tf.reduce_sum(mask,axis=0)+epsilon)
        batch_var = tf.reduce_sum(tf.square(x-batch_mean)*mask,axis=0)/(tf.reduce_sum(mask,axis=0)+epsilon)

        # create decay
        decay_tensor1 = tf.ones((size_batch,1))*decay
        decay_tensor2 = tf.ones((size-size_batch,1))
        decay_tensor = tf.concat((decay_tensor1,decay_tensor2),axis=0)
        # pad batch
        batch_mean_pad = tf.pad(batch_mean, paddings=tf.constant([(0, 1,), (0, 0)]) * (size - size_batch))
        batch_var_pad = tf.pad(batch_var, paddings=tf.constant([(0, 1,), (0, 0)]) * (size - size_batch))

        train_mean_op = tf.assign(pop_mean, pop_mean * decay_tensor + batch_mean_pad * (1 - decay))
        train_var_op = tf.assign(pop_var, pop_var * decay + batch_var_pad * (1 - decay))

        def batch_statistics():
            with tf.control_dependencies([train_mean_op, train_var_op]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, offset_batch, scale_batch, epsilon)

        def population_statistics():
            return tf.nn.batch_normalization(x, pop_mean_batch, pop_var_batch, offset_batch, scale_batch, epsilon)

        return tf.cond(training, batch_statistics, population_statistics)

# def assign_one()
#
# def batch_norm_3d_new(x, training, mask=None, name='bn', epsilon=1e-3, decay=0.999, size=None):
#     '''customized batch_norm with mask
#         default: x=B*T*D, mask=B*T
#     '''
#
#     with tf.variable_scope(name):
#         if size is None:
#             size = x.get_shape().as_list()[1]
#
#         scale = [tf.get_variable('scale'+str(i), [1,x.get_shape()[-1]], initializer=tf.constant_initializer(0.1)) for i in range(size)]
#         offset = [tf.get_variable('offset'+str(i), [1,x.get_shape()[-1]]) for i in range(size)]
#
#         pop_mean = [tf.get_variable('pop_mean'+str(i), [1,x.get_shape()[-1]], initializer=tf.zeros_initializer, trainable=False) for i in range(size)]
#         pop_var = [tf.get_variable('pop_var'+str(i), [1,x.get_shape()[-1]], initializer=tf.ones_initializer, trainable=False) for i in range(size)]
#         # batch_mean, batch_var = tf.nn.moments(x, [0])
#         # mask = tf.pad(mask,paddings=tf.constant([(0, 0,), (0, 1)]) * (size-tf.shape(mask)[1]))
#         mask = tf.expand_dims(mask,axis=-1)
#         x *= mask
#         batch_mean = tf.reduce_sum(x,axis=0)/(tf.reduce_sum(mask,axis=0)+epsilon)
#         batch_var = tf.reduce_sum(tf.square(x-batch_mean)*mask,axis=0)/(tf.reduce_sum(mask,axis=0)+epsilon)
#
#         train_mean_op = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
#         train_var_op = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
#
#         def batch_statistics():
#             with tf.control_dependencies([train_mean_op, train_var_op]):
#                 return tf.nn.batch_normalization(x, batch_mean, batch_var, offset, scale, epsilon)
#
#         def population_statistics():
#             return tf.nn.batch_normalization(x, pop_mean, pop_var, offset, scale, epsilon)
#
#         return tf.cond(training, batch_statistics, population_statistics)