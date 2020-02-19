from argparse import ArgumentParser

def make_args():
    parser = ArgumentParser()
    # dataset name
    parser.add_argument('--dataset', dest='dataset', default='xing',
                        help='dataset name')
    # for a normal boolean arg, add it means set it to True, and default is False
    # for those with --.*_no, add it means set it to False, and default is True
    parser.add_argument('--warm_start', dest='warm_start',action='store_true',
                        help='whether use warm start train/test')
    ### Training Parameters ###
    parser.add_argument('--lr', dest='learning_rate',default=1e-2,
                        help='learning rate')
    parser.add_argument('--lr_schedule_no', dest='lr_schedule',action='store_false',
                        help='change the learning rate while training')
    parser.add_argument('--epoch_max', dest='epoch_max',default=200,
                        help='how many epoches to train')
    parser.add_argument('--batch_size', dest='batch_size',default=32,
                        help='batch size')  #if args.model_type in {'max_pin', 'mv'}:args.batch_size = 8
    # normalization
    parser.add_argument('--has_batchnorm', dest='has_batchnorm',action='store_true',
                        help='does not support single tcn')
    parser.add_argument('--has_weightnorm', dest='has_weightnorm',action='store_true',
                        help='does not support single tcn')
    parser.add_argument('--has_layernorm', dest='has_layernorm',action='store_true',
                        help='should be False!! (will look future events in advance)')
    ### Network Parameters ###
    parser.add_argument('--model_type', dest='model_type',default='mv_xing',
                        help='high-level model type: mv: moving average model (single level);'+\
                            'gru: gru model (single level);'+\
                            'tcn : tcn model (single level);'+\
                            'max_pin: max pin model (single level);'+\
                            'hier: default hierachical model, high level use gru;'+\
                            'hier_res: hierachical model, high level use mlp+residual connection;'+\
                            'hier_low: sanity check, hierachical structure but disable high level model;'+\
                            'hier_baseline: reproducing the structure of the baseline paper;'+\
                            'hier_slice: use pool of pin in last session, and the topic info for the input to the low level model'
                        )
    parser.add_argument('--model_low_type', dest='model_low_type',default='gru',
                        help='low-level model type: mv: moving average model;'+\
                            'gru: gru model;'+\
                            'tcn : tcn model;'+\
                            'tcn_multi: make multiple prediction, not recommend to use;'+\
                            'tcn_gmm: tcn with gaussian mixture model'
                        )
    parser.add_argument('--item_num', dest='item_num', default=20777+1,
                        help='item_num for xing dataset, plus null')
    # Making all model have ~1.3M trainable parameters
    # [for gru]
    parser.add_argument('--hidden_dim', dest='hidden_dim',default=128,
                        help='hidden state size for gru') #if args.model_type == 'gru':args.hidden_dim = 200
    parser.add_argument('--num_layer', dest='num_layer',default=2,
                        help='num layer for gru')
    # [for tcn]
    parser.add_argument('--tcn_channel', dest='tcn_channel',default=[128, 128],
                        help='output size for each hidden layer') # if args.model_type == 'tcn':args.tcn_channel = [128, 128, 128, 128, 256, 256]
    parser.add_argument('--kernel_size', dest='kernel_size',default=5,
                        help='convolution kernel size')
    parser.add_argument('--strides', dest='strides',default=1,
                        help='convolution strides')
    parser.add_argument('--multi_num', dest='multi_num',default=3,
                        help='for tcn_multi: how many prediction to make')
    parser.add_argument('--dropout', dest='dropout',default=0.0, type=float,
                        help='for tcn, 0.0 means no dropout')
    # [for hier_slice]
    parser.add_argument('--topic_dim', dest='topic_dim',default=50,
                        help='num topics in hier_slice')
    # [for tcn_gmm]
    parser.add_argument('--sigma', dest='sigma',default=0.1,
                        help='None -- learn sigma of gmm; 0.1--set fixed sigma')
    parser.add_argument('--num_mixture', dest='num_mixture',default=5,
                        help='number of gaussian to use')
    parser.add_argument('--pred_mode', dest='pred_mode',default='gmm_score',
                        help='pred_mode: gmm_score, gmm_max_mean, gmm_mean'+\
                        'gmm_socre: only use score to rank pins;'+\
                        'gmm_max_mean: use the max of gaussian mean to predict;'+\
                        'gmm_mean: use the weighted avg of gaussian mean to predict')
    parser.add_argument('--mix_weight', dest='mix_weight',default=0.01,
                        help='weight for mle in mix_hinge loss')
    # baseline model parameters
    parser.add_argument('--mv_isWeight', dest='mv_isWeight',action='store_true',
                        help='for mv, if weighted sum')
    parser.add_argument('--lag', dest='lag',default=10,
                        help='for var, mv, keep how many history activities')
    ### Feature Parameters ###
    parser.add_argument('--has_impression', dest='has_impression',action='store_true',
                        help='if concat input with impression data')
    parser.add_argument('--has_gap', dest='has_gap',action='store_true',
                        help='if consider time gap decay between sessions')
    parser.add_argument('--gap_bandwidth', dest='gap_bandwidth',default=168,
                        help='bandwidth, at which point the vector will decay -> 0.3')
    parser.add_argument('--train_gap', dest='train_gap',action='store_true',
                        help='if train how to decay with gap, otherwise use predefined decay scheme')
    parser.add_argument('--data_noise', dest='data_noise',default=None,
                        help='default None, add noise to x, data augmentation')
    parser.add_argument('--binary_input', dest='binary_input',action='store_true',
                        help='default False, if make input binary, better turn off')
    parser.add_argument('--input_dim', dest='input_dim',default=512,
                        help='dim of x')
    parser.add_argument('--output_dim', dest='output_dim',default=20777+1,
                        help='dim of y')

    ### Loss Parameters ###
    parser.add_argument('--loss', dest='loss',default='cross_entropy',
                        help='type of loss: l2, nce,'+\
                        'hinge_linear, hinge_sigmoid, hinge_logsigmoid, bpr,'+\
                        'gmm_prob, gmm_prob_max, gmm_prob_hinge, gmm_hinge_mix, gmm_prob_hinge_logsigmoid')
    parser.add_argument('--rank_metric', dest='rank_metric',default='l2',
                        help='distance metric used for evaluation ranking,'+\
                        'l2, inner_prod, loss (use loss function to rank)')
    parser.add_argument('--num_neg_sample', dest='num_neg_sample',default=20,
                        help='num_neg_sample used')
    parser.add_argument('--nce_weight', dest='nce_weight',default=1,
                        help='for nce weight')
    parser.add_argument('--l2_normalize', dest='l2_normalize',action='store_true',
                        help='where normalize output, default False')
    parser.add_argument('--hinge_delta', dest='hinge_delta',default=0.1,
                        help='delta in hinge loss')

    ### Data Parameters ###
    parser.add_argument('--max_seq_len', dest='max_seq_len',default=500,
                        help='when loading user data, maximum activity counts for a user')
    parser.add_argument('--max_impression_len', dest='max_impression_len',default=20,
                        help='maximum impression for a feedview')
    parser.add_argument('--max_activity_len', dest='max_activity_len',default=20,
                        help='[for hier models] max activities within a session')
    parser.add_argument('--max_session_num', dest='max_session_num',default=10,
                        help='max_session_num per batch')
    parser.add_argument('--epoch_batches_train', dest='epoch_batches_train',default=500,
                        help='how many batches to train in an socalled epoch, '+\
                             'since training a real epoch takes too long time')
    parser.add_argument('--load_worker_num', dest='load_worker_num',default=8,
                        help='how many workers when preparing data, not for get_batch function')
    # I/O Parameters
    # these parameters are frequenly changed in different settings,
    # thus both store_true and store_false are implemented
    parser.add_argument('--save', dest='save',action='store_true',
                        help='whether save or not')
    parser.add_argument('--save_no', dest='save',action='store_false',
                        help='whether save or not')
    parser.add_argument('--save_epoch', dest='save_epoch',default=20,
                        help='save every k epoch')
    parser.add_argument('--test_epoch', dest='test_epoch',default=20, type=int,
                        help='test every k epoch')
    parser.add_argument('--load', dest='load',action='store_true',
                        help='whether load pretrained model or not')
    parser.add_argument('--load_no', dest='load',action='store_false',
                        help='whether load pretrained model or not')
    parser.add_argument('--load_epoch', dest='load_epoch',default=20,
                        help='the epoch to load')
    parser.add_argument('--train', dest='train',action='store_true',
                        help='whether train the model or not, default True')
    parser.add_argument('--train_no', dest='train',action='store_false',
                        help='whether train the model or not, default True')
    parser.add_argument('--validate', dest='validate', action='store_true',
                        help='whether validate the model or not, default True')
    parser.add_argument('--validate_no', dest='validate', action='store_false',
                        help='whether validate the model or not, default True')
    parser.add_argument('--test', dest='test', action='store_true',
                        help='whether test the model or not, default True')
    parser.add_argument('--test_no', dest='test', action='store_false',
                        help='whether test the model or not, default True')
    parser.add_argument('--shuffle', dest='shuffle', action='store_true',
                        help='whether shuffle the data or not, default True')
    parser.add_argument('--shuffle_no', dest='shuffle', action='store_false',
                        help='whether shuffle the data or not, default True')
    parser.add_argument('--html', dest='html',action='store_true',
                        help='whether output html visualization for model performance')
    parser.add_argument('--html_no', dest='html',action='store_false',
                        help='whether output html visualization for model performance')
    parser.add_argument('--test_random', dest='test_random',action='store_true',
                        help='test on random impression')
    parser.add_argument('--test_random_no', dest='test_random',action='store_false',
                        help='test on random impression')
    parser.add_argument('--data_type', dest='data_type',default='1m',
                        help='1m, 10m, 1m_splittime')
    parser.add_argument('--filter_user', dest='filter_user',action='store_true',
                        help='whether select US Female users from dataset')
    parser.add_argument('--filter_user_no', dest='filter_user',action='store_false',
                        help='whether select US Female users from dataset')
    parser.add_argument('--load_num_train', dest='load_num_train',default=800,
                        help='how many train data partitions to load, changing this will cause re-caching data')
    parser.add_argument('--load_num_test', dest='load_num_test',default=800,
                        help='how many test data partitions to load, changing this will cause re-caching data')
    parser.add_argument('--test_data_ratio', dest='test_data_ratio',default=1,
                        help='default 1, the propotion of test data used')
    parser.add_argument('--data_splits', dest='data_splits',default=100,
                        help='when caching data, how many splits to save. default 100')


    # eval
    parser.set_defaults(save=False,load=False,train=False,validate=False,test=True,html=False,test_random=False,filter_user=False, shuffle=False)
    # train and eval
    # parser.set_defaults(save=True,load=False,train=True,validate=False,test=True,html=False,test_random=True,filter_user=False, shuffle=True)
    # plot
    # parser.set_defaults(save=False, load=True, train=False, validate=False, test=False, html=True, test_random=False,filter_user=False, shuffle=False)
    # plot_mv
    # parser.set_defaults(save=False, load=False, train=False, validate=False, test=False, html=True, test_random=False,filter_user=False, shuffle=False)

    # parser.set_defaults(save=True,load=False,train=True,validate=False,test=True,html=False,test_random=False,filter_user=False)

    args = parser.parse_args()
    args = args_adjust(args)
    print('args: ' + args.name)
    return args

def args_make_input_path(args):
    # Data path
    if args.data_type == '1m':
        args.train_path = '/data1/home/ywang/user_data_homefeed_1m_train_sort_session/'
        args.test_path = '/data1/home/ywang/user_data_homefeed_1m_test_sort_session/'
        if args.filter_user:
            args.data_splits = 20
    if args.data_type == '1m_splittime':
        args.train_path = '/data1/home/ywang/user_data_homefeed_1m_train_sort_session_splittime/'
        args.test_path = '/data1/home/ywang/user_data_homefeed_1m_test_sort_session_splittime/'
        if args.filter_user:
            args.data_splits = 20
    elif args.data_type == '10m':
        args.train_path = '/data1/home/ywang/user_data_homefeed_10m_train_sort_session/'
        args.test_path = '/data1/home/ywang/user_data_homefeed_10m_test_sort_session/'
        if not args.filter_user:
            args.load_worker_num = 1
            args.data_splits = 800
    args.train_cache_path = args.train_path
    args.test_cache_path = args.test_path
    return args

def args_make_output_path(args):
    # the name used for output
    args.name = '1030_final' + args.model_type + args.model_low_type + args.loss + 'hinge_delta' + str(
        args.hinge_delta) + 'gap' + str(args.has_gap) + 'bandwidth' + str(args.gap_bandwidth) + str(args.train_gap) \
                + 'impression' + str(args.has_impression) + 'noise' + str(
        args.data_noise) + 'rank_metric' + args.rank_metric \
                + 'dropout' + str(args.dropout) + 'batchnorm' + str(args.has_batchnorm) + 'weightnorm' + str(
        args.has_weightnorm) \
                + 'kernel' + str(args.kernel_size) + 'channel' + ','.join(str(i) for i in args.tcn_channel) \
                + 'data' + args.data_type + 'loadnum' + str(args.load_num_train) + 'split' + str(
        args.data_splits) + 'filter' + str(args.filter_user)

    # an example name to load
    # hiertcn
    # args.name_load = '0906_finalhiertcnhinge_logsigmoidhinge_delta0.1gapFalsebandwidth168FalseimpressionFalsenoiseNonerank_metricl2dropout0batchnormFalseweightnormFalsekernel5channel128,128,128,128data10mloadnum800split800filterFalse'
    # hiergru
    # args.name_load = '0906_finalhiergruhinge_logsigmoidhinge_delta0.1gapFalsebandwidth168FalseimpressionFalsenoiseNonerank_metricl2dropout0batchnormFalseweightnormFalsekernel5channel128,128,128,128data10mloadnum800split800filterFalse'
    # hiermv
    # args.name_load = '0906_finalhiermvhinge_logsigmoidhinge_delta0.1gapFalsebandwidth168FalseimpressionFalsenoiseNonerank_metricl2dropout0batchnormFalseweightnormFalsekernel5channel128,128,128,128data10mloadnum800split800filterFalse'
    # hier_slice tcn
    # args.name_load = '0906_final_statehier_slicetcnhinge_logsigmoidhinge_delta0.1gapFalsebandwidth168FalseimpressionFalsenoiseNonerank_metricl2dropout0batchnormFalseweightnormFalsekernel5channel128,128,128,128data10mloadnum800split800filterFalse'
    # hier_baseline gru
    # args.name_load = '0909_finalhier_baselinegruhinge_logsigmoidhinge_delta0.1gapFalsebandwidth168FalseimpressionFalsenoiseNonerank_metricl2dropout0batchnormFalseweightnormFalsekernel5channel128,128,128,128data10mloadnum800split800filterFalse'

    # gru
    # args.name_load = '0906_finalgrugruhinge_logsigmoidhinge_delta0.1gapFalsebandwidth168FalseimpressionFalsenoiseNonerank_metricl2dropout0batchnormFalseweightnormFalsekernel5channel128,128,128,128data10mloadnum800split800filterFalse'
    # tcn
    # args.name_load = '0906_finaltcntcnhinge_logsigmoidhinge_delta0.1gapFalsebandwidth168FalseimpressionFalsenoiseNonerank_metricl2dropout0batchnormFalseweightnormFalsekernel5channel128,128,128,128,256,256data10mloadnum800split800filterFalse'

    # args.name_load = '0911_finalhiertcn_gmmgmm_prob_hinge_logsigmoidhinge_delta0.1gapFalsebandwidth168FalseimpressionFalsenoiseNonerank_metricl2dropout0batchnormFalseweightnormFalsekernel5channel128,128,128,128data10mloadnum800split800filterFalse'


    # warm
    # hier_slice
    # args.name_load = '0913_finalhier_slicetcnhinge_logsigmoidhinge_delta0.1gapFalsebandwidth168FalseimpressionFalsenoiseNonerank_metricl2dropout0.0batchnormFalseweightnormFalsekernel5channel128,128,128,128data1mloadnum800split100filterFalsewarm'
    # hier tcn
    # args.name_load = '0913_finalhiertcnhinge_logsigmoidhinge_delta0.1gapFalsebandwidth168FalseimpressionFalsenoiseNonerank_metricl2dropout0.0batchnormFalseweightnormFalsekernel5channel128,128,128,128data1mloadnum800split100filterFalsewarm'
    # hier baseline
    # args.name_load = ''
    # tcntcn
    # args.name_load = '0913_finalvisualtcntcnbprhinge_delta0.1gapFalsebandwidth168FalseimpressionFalsenoiseNonerank_metricl2dropout0.0batchnormFalseweightnormFalsekernel5channel128,128,128,128,256,256data1mloadnum800split100filterFalsewarm'
    # compare_loss
    # args.name_load = '0913_finalcomparelosshiertcnhinge_logsigmoidhinge_delta0.1gapFalsebandwidth168FalseimpressionFalsenoiseNonerank_metricl2dropout0.0batchnormFalseweightnormFalsekernel5channel128,128,128,128data1mloadnum800split100filterFalse'


    # 1
    # args.name_load = '0913_finalcomparelosshiertcnl2hinge_delta0.1gapFalsebandwidth168FalseimpressionFalsenoiseNonerank_metricl2dropout0.0batchnormFalseweightnormFalsekernel5channel128,128,128,128data1mloadnum800split100filterFalse'
    # 2
    # args.name_load = '0913_finalcomparelosshiertcnncehinge_delta0.1gapFalsebandwidth168FalseimpressionFalsenoiseNonerank_metricl2dropout0.0batchnormFalseweightnormFalsekernel5channel128,128,128,128data1mloadnum800split100filterFalse'
    # args.name_load = '0913_finalcomparelosshiertcnbprhinge_delta0.1gapFalsebandwidth168FalseimpressionFalsenoiseNonerank_metricl2dropout0.0batchnormFalseweightnormFalsekernel5channel128,128,128,128data1mloadnum800split100filterFalse'
    args.name_load = '0913_finalcomparelosshiertcnhinge_linearhinge_delta0.1gapFalsebandwidth168FalseimpressionFalsenoiseNonerank_metricl2dropout0.0batchnormFalseweightnormFalsekernel5channel128,128,128,128data1mloadnum800split100filterFalse'
    # args.name_load = '0913_finalcomparelosshiertcnhinge_logsigmoidhinge_delta0.1gapFalsebandwidth168FalseimpressionFalsenoiseNonerank_metricl2dropout0.0batchnormFalseweightnormFalsekernel5channel128,128,128,128data1mloadnum800split100filterFalse'


    if args.warm_start:
        args.name += 'warm'
    # username = 'jyou'
    args.log_path = 'run/tensorboard/' + args.name + '/'
    args.html_path = 'run/html/'
    args.save_path = 'run/model/' + args.name + '/'
    args.load_path = 'run/model/' + args.name_load + '/'
    return args

def args_adjust(args):
    '''
    adjust args for some specific models
    :param args: args
    :return:
    '''

    # deprecated, will remove in the final version
    # for rule based model
    # if args.model_type in {'max_pin', 'mv'}:
    #     args.batch_size = 8

    # for single level gru
    if args.model_type == 'gru':
        args.hidden_dim = 200

    # for single level tcn
    if args.model_type == 'tcn':
        args.tcn_channel = [128, 128, 128, 128, 256, 256]

    args = args_make_input_path(args)
    args = args_make_output_path(args)

    return args

args = make_args()
