from data_loader import *
from model import *
from run_rule_xing import *
from run_nn_xing import *
from run_hier_xing import *
import os
import time


while os.path.isdir(args.log_path):
    args.log_path = args.log_path[:-1]+'1/'
    print('logdir exist, new logdir: ', args.log_path)
os.makedirs(args.log_path)

# prepare dataset
loader_test_random = None # optional
visual_batch = None # optional
if 'hier' not in args.model_type:
    if not args.warm_start:
        loader_train = Dataloader_single_level_model_xing(args,'train')
        loader_validate = Dataloader_single_level_model_xing(args,'validate')
        loader_test = Dataloader_single_level_model_xing(args,'test')
    else:
        loader_train = Dataloader_single_level_model_xing(args, 'train')
        loader_validate = Dataloader_single_level_model_xing(args, 'train')
        loader_test = Dataloader_single_level_model_xing(args, 'train')
else:
    if not args.warm_start:
        loader_train = Dataloader_hier_model_xing(args,'train')
        loader_validate = Dataloader_hier_model_xing(args,'validate')
        loader_test = Dataloader_hier_model_xing(args,'test')
    else:
        loader_train = Dataloader_hier_model_xing(args, 'train')
        loader_validate = Dataloader_hier_model_xing(args, 'train')
        loader_test = Dataloader_hier_model_xing(args, 'train')


if args.model_type in {'gru','tcn','gru_gmm','tcn_gmm','gru_gap_weight','gru_gap_weight_v2','mv_xing'}:
    run_nn(args,loader_train,loader_validate,loader_test,warm_start=args.warm_start)
elif args.model_type in {'mv','max_pin'}:
    run_rule(args,loader_test,warm_start=args.warm_start)
elif args.model_type in {'hier','hier_res','hier_low','hier_baseline','hier_slice'}:
    run_hier(args,loader_train,loader_validate,loader_test,warm_start=args.warm_start)
else:
    raise NotImplementedError