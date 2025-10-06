import argparse
import os
import torch
import numpy as np
import random

from exp.exp_main import Exp_Main
import warnings
warnings.filterwarnings('ignore')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='MoLE (Mixture-of-Linear-Experts)')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='MoLE_DLinear',
                    help='model name, options: [MoLE_DLinear, MoLE_RLinear, MoLE_RMLP, MoLE_TreeNN, MoLE_XGBoost]')

# data loader
parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='electricity.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# MoLE specific
parser.add_argument('--t_dim', type=int, default=4, help='number of temporal experts')
parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')
parser.add_argument('--channel', type=int, default=321, help='number of channels')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout rate')
parser.add_argument('--disable_rev', action='store_true', default=False, help='disable reversible instance normalization')
parser.add_argument('--drop', type=float, default=0.05, help='dropout rate')

# TreeNN specific
parser.add_argument('--tree_depth', type=int, default=3, help='depth of each decision tree')
parser.add_argument('--num_trees', type=int, default=2, help='number of trees in the ensemble')

# Augmentation
parser.add_argument('--in_batch_augmentation', action='store_true', default=False, help='in batch augmentation')
parser.add_argument('--in_dataset_augmentation', action='store_true', default=False, help='in dataset augmentation')
parser.add_argument('--aug_method', type=str, default='f_mask', help='augmentation method')
parser.add_argument('--aug_rate', type=float, default=0.5, help='augmentation rate')
parser.add_argument('--aug_data_size', type=int, default=1, help='augmentation data size')
parser.add_argument('--wo_original_set', action='store_true', default=False, help='without original dataset')
parser.add_argument('--closer_data_aug_more', action='store_true', default=False, help='closer data augmentation more')

# Other settings
parser.add_argument('--data_size', type=float, default=1.0, help='data size ratio')
parser.add_argument('--test_time_train', action='store_true', default=False, help='test time training')
parser.add_argument('--save_gating_weights', type=str, default=None, help='save gating weights')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# CPU testing
parser.add_argument('--use_cpu', action='store_true', default=False, help='use CPU instead of GPU')
parser.add_argument('--show_num_parameters_only', action='store_true', default=False, help='show number of parameters only')

args = parser.parse_args()

# Set random seeds
def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

fix_seed(args.seed)

# Override GPU settings if use_cpu is specified
if args.use_cpu:
    args.use_gpu = False
    print("Using CPU for computation")

# Set default values for enc_in based on dataset
if args.data_path == 'electricity.csv' or args.data_path == 'ECL.csv':
    if args.enc_in == 7:  # default value, override it
        args.enc_in = 321
        args.dec_in = 321
        args.c_out = 321

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii)

        exp = Exp(args)  # set experiments
        
        # Show number of parameters if requested
        if args.show_num_parameters_only:
            total_params = sum(p.numel() for p in exp.model.parameters())
            trainable_params = sum(p.numel() for p in exp.model.parameters() if p.requires_grad)
            print(f'Total parameters: {total_params:,}')
            print(f'Trainable parameters: {trainable_params:,}')
            exit()
        
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()