from args import *
from model_gru import *
from model_gru_gmm import *
from model_tcn import *
from model_tcn_gmm import *
from model_var import *
from model_mv import *
from model_max_pin import *

from model_hier import *
from model_hier_res import *
from model_hier_low import *
from model_hier_baseline import *
from model_hier_slice import *
from loss import *

## Build prediction model
# Input
x = tf.placeholder(dtype=tf.float32, shape=[None, None, args.input_dim], name='x') #[batch,T,feature_dim]
y = tf.placeholder(dtype=tf.float32, shape=[None, None, args.input_dim], name='y')
x_gap = tf.placeholder(dtype=tf.float32, shape=[None, None, 1], name='x_gap') #[batch,T,1]

y_pred = tf.placeholder(dtype=tf.float32, shape=[None, None, args.input_dim], name='y_pred')

x_impression = tf.placeholder(dtype=tf.float32, shape=[None, None, None,args.input_dim], name='x_impression') # [batch,T,impression_num,feature_dim]
y_impression = tf.placeholder(dtype=tf.float32, shape=[None, None, None,args.output_dim], name='y_impression') # [batch,T,impression_num,feature_dim]
x_var = tf.placeholder(dtype=tf.float32, shape=[None, None, args.lag, args.input_dim], name='x_var') #[batch,T,lag,feature_dim]

# session based model
x_list = [tf.placeholder(dtype=tf.float32, shape=[None, None, args.input_dim]) for _ in range(args.max_session_num)]
x_impression_list = [tf.placeholder(dtype=tf.float32, shape=[None, None, None, args.input_dim]) for _ in range(args.max_session_num)]
x_gap_list = [tf.placeholder(dtype=tf.float32, shape=[None, 1]) for _ in range(args.max_session_num)]
y_list = [tf.placeholder(dtype=tf.float32, shape=[None, None, args.input_dim]) for _ in range(args.max_session_num)]
state_in = tf.placeholder(dtype=tf.float32, shape=[None,args.hidden_dim*args.num_layer],name='state_in')

y_slice_in = tf.placeholder(dtype=tf.float32, shape=[None,args.output_dim],name='y_slice_in')
topic_state_in = tf.placeholder(dtype=tf.float32, shape=[None,args.topic_dim],name='topic_state_in')

mask_list = [tf.placeholder(dtype=tf.float32, shape=[None, 1]) for _ in range(args.max_session_num)]
mask_warmstart = tf.placeholder(dtype=tf.float32, shape=[None, None], name='mask_warmstart')

lr = tf.placeholder(dtype=tf.float32, shape=None, name='lr')
is_train = tf.placeholder(dtype=bool)


# for xing data
x_id = tf.placeholder(dtype=tf.int32, shape=[None, None], name='x_id') #[batch,T,feature_dim]
y_id = tf.placeholder(dtype=tf.int32, shape=[None, None], name='y_id')
y_pred_id = tf.placeholder(dtype=tf.int32, shape=[None, None], name='y_pred_id')
x_id_list = [tf.placeholder(dtype=tf.int32, shape=[None, None]) for _ in range(args.max_session_num)]
y_id_list = [tf.placeholder(dtype=tf.int32, shape=[None, None]) for _ in range(args.max_session_num)]
y_id_slice_in = tf.placeholder(dtype=tf.int32, shape=[None,args.output_dim],name='y_slice_in')

if args.dataset=='xing':
    # mask_x = tf.sign(x)  # b,t
    x = tf.one_hot(x_id,depth=args.item_num)*tf.cast(tf.expand_dims(tf.sign(x_id), axis=-1),dtype=tf.float32)
    y = tf.one_hot(y_id,depth=args.item_num)*tf.cast(tf.expand_dims(tf.sign(y_id), axis=-1),dtype=tf.float32)
    y_pred = tf.one_hot(y_pred_id,depth=args.item_num)*tf.cast(tf.expand_dims(tf.sign(y_pred_id), axis=-1),dtype=tf.float32)
    x_list = [tf.one_hot(temp,depth=args.item_num)*tf.cast(tf.expand_dims(tf.sign(temp), axis=-1),dtype=tf.float32) for temp in x_id_list]
    y_list = [tf.one_hot(temp, depth=args.item_num) * tf.cast(tf.expand_dims(tf.sign(temp), axis=-1), dtype=tf.float32) for
          temp in y_id_list]
    mask_y = tf.cast(tf.sign(y_id),dtype=tf.float32) # b,t
    print(mask_y.get_shape())


# set up model
if args.model_type == 'gru':
    pred = model_gru(args,x, x_impression)
elif args.model_type == 'mv_xing':
    pred = model_mv_tf(x,n=args.lag)
elif args.model_type == 'tcn':
    mask = tf.sign(tf.reduce_sum(tf.abs(x), 2))
    pred = model_tcn(args,x,x_impression=x_impression, mask=mask, training=is_train)
elif args.model_type == 'tcn_gmm':
    pred = model_tcn_gmm(args, x,x_impression,training=is_train)
elif args.model_type == 'gru_gmm':
    pred = model_gru_gmm(args, x,x_impression)
elif args.model_type == 'gru_gap_weight':
    pred = model_gru_gap_weight(args, x)
elif args.model_type in {'mv','mv_gap', 'mv_sess', 'max_pin'}:
    pred = y_pred
elif args.model_type == 'var':
    pred = model_var(x_var)
elif args.model_type == 'hier':
    pred, state = model_hier(args,x_list,y_list,mask_list,state_in,x_gap_list,x_impression_list,training=is_train)
elif args.model_type == 'hier_baseline':
    pred, state = model_hier_baseline(args,x_list,y_list,mask_list,state_in,x_gap_list,x_impression_list,training=is_train)
elif args.model_type == 'hier_slice':
    pred, state, y_slice, topic_state,topic_state_list = model_hier_slice(args,x_list,y_list,mask_list,state_in,y_slice_in,topic_state_in,x_gap_list,x_impression_list,training=is_train)
elif args.model_type == 'hier_low':
    pred = model_hier_low(args,x_list)
    state = state_in
elif args.model_type == 'hier_res':
    pred, state = model_hier_res(args,x_list, y_list, mask_list, state_in)
else:
    raise NotImplementedError

## calc loss

# compute mask
# mask_y = tf.sign(tf.reduce_sum(tf.abs(y), 2)) # b,t
if args.warm_start:
    mask_y *= mask_warmstart
# mask the prediction
pred *= tf.expand_dims(mask_y,-1)
# pred = tf.nn.l2_normalize(pred, dim=-1)

loss = calc_loss(pred,y)

# apply mask to loss
loss *= mask_y
activity_count = tf.reduce_sum(mask_y, 1)
user_count = tf.reduce_sum(tf.sign(activity_count),axis=-1)
activity_count += tf.constant(1e-6)

loss = tf.reduce_sum(loss,1) / activity_count # B
loss = tf.reduce_sum(loss)/user_count

# compute the impression mask, 0 means no impression
# mask_impression = tf.sign(tf.reduce_sum(tf.abs(y_impression),axis=-1)) # b,t,k
# mask_impression_bias = tf.cast(tf.logical_not(tf.cast(mask_impression,tf.bool)),tf.float32)*-100 # b,t,k

## calc score

# score = calc_score(pred, y_impression)+mask_impression_bias

## calc metric
# recall1_mean_user, recall5_mean_user, recall10_mean_user, reciprocal_ranks_float_mean_user, ranks_float_mean_user, ranks_float, ranks_indices,ranks_values,ranks =\
#     calc_metric(pred,mask_y,activity_count,user_count,y_id)

recall1_mean_user, recall5_mean_user, recall10_mean_user, reciprocal_ranks_float_mean_user, ranks_float_mean_user, ranks_float,ranks =\
    calc_metric_fast(pred,mask_y,activity_count,user_count,y)

## set up train optimizer
if args.model_type not in {'mv','mv_gap', 'max_pin','mv_xing'}:
    if args.has_batchnorm:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    else:
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
init = tf.global_variables_initializer()


## save
if args.model_type not in ['var','max_pin','mv','mv_xing']:
    saver = tf.train.Saver()

## create placeholder for visualization
ranks_analysis_placeholder = tf.placeholder(dtype=tf.uint8,shape=[None,None,None,4])
emb_placeholder = tf.placeholder(dtype=tf.uint8,shape=[None,None,None,4])

## set up tensorboard
tf.summary.scalar('loss', loss)
tf.summary.scalar('lr', lr)
# variable_summaries(pred,'pred')
# variable_summaries(y,'real')
# if 'hier' in args.model_type:
#     variable_summaries(state,'state')
train_summary = tf.summary.merge_all()
val_summary = tf.summary.merge_all()

## set up visualization in tensorboard
ranks_analysis_plot_summary = tf.summary.image("ranks_analysis_plot", ranks_analysis_placeholder, max_outputs=6)
emb_plot_summary = tf.summary.image("emb_plot", emb_placeholder, max_outputs=4)
plot_summary = tf.summary.merge([emb_plot_summary,ranks_analysis_plot_summary])

# print parameter count
total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    # print(variable.name)
    # print(shape)
    # print(len(shape))
    variable_parameters = 1
    for dim in shape:
        # print(dim)
        variable_parameters *= dim.value
    # print(variable_parameters)
    total_parameters += variable_parameters
print('total parameters count:',total_parameters)
