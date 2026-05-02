import argparse
import os
import torch
from exp import exp_main
import random
import numpy as np

fix_seed = 2021
# fix_seed = 2022
# fix_seed = 2023
# fix_seed = 2024
# fix_seed = 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='MFIKAN')

# basic config
# General parameters
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='MFIKAN',
                    help='model name, options: [MFIKAN, Autoformer, Transformer, TimesNet, KAN, xPatch]')
parser.add_argument('--Exp', type=str, required=False, default='exp_main',
                    help='Exp name, options: [exp_main]')
parser.add_argument('--train_only', type=bool, required=False, default=False, help='perform training on full input dataset without validation and testing')
parser.add_argument('--donot_save', type=int, required=False, default=0, help='do not save checkpoint')
parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

# Data loader parameters
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')

# Forecasting task parameters
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')

# Inputation and Anomaly Detection parameters
parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio for imputation')
parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%)')

# Model-specific parameters
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--c_in', type=int, default=7, help='input size')
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
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--channel_independence', type=int, default=1,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default=None,
                    help='down sampling method, only support avg, max, conv')
parser.add_argument('--seg_len', type=int, default=96,
                    help='the length of segmen-wise iteration of SegRNN')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', action='store_true', default=True,
                    help='affine parameter for RevIN normalization')

# VMD-KAN specific parameters
parser.add_argument('--vmd_scales', type=int, default=5,
                    help='number of VMD decomposition scales (K)')
parser.add_argument('--vmd_alpha', type=float, default=2000,
                    help='VMD alpha parameter for mode separation')
parser.add_argument('--vmd_tau', type=float, default=0.0,
                    help='VMD tau parameter for noise tolerance')
parser.add_argument('--vmd_dc', type=int, default=0,
                    help='VMD DC mode (0 or 1)')
parser.add_argument('--vmd_tol', type=float, default=1e-6,
                    help='VMD convergence tolerance')
parser.add_argument('--vmd_n_iter', type=int, default=500,
                    help='VMD maximum iterations')
parser.add_argument('--vmd_init', type=str, default='uniform',
                    help='VMD initialization method: [uniform, log, random]')
parser.add_argument('--vmd_chunk_size', type=int, default=32,
                    help='batch size for VMD processing')

# Trend-Season decomposition parameters
parser.add_argument('--use_stl', action='store_true', default=False,
                    help='use STL decomposition instead of moving average')
parser.add_argument('--stl_period', type=int, default=24,
                    help='period for STL decomposition')
parser.add_argument('--stl_seasonal', type=int, default=13,
                    help='seasonal parameter for STL')
parser.add_argument('--stl_trend', type=int, default=None,
                    help='trend parameter for STL')
parser.add_argument('--use_ts_feature_norm', action='store_true', default=True,
                    help='use feature normalization in trend-season decomposition')

# Multi-scale prediction parameters
parser.add_argument('--num_pred_scales', type=int, default=3,
                    help='number of prediction scales in VMD-KAN')
parser.add_argument('--use_scale_attention', action='store_true', default=True,
                    help='use attention mechanism for scale weighting')
parser.add_argument('--use_scale_norm', action='store_true', default=False,
                    help='use normalization after VMD decomposition')

# KAN encoder/decoder specific parameters
parser.add_argument('--kan_encoder_layers', type=int, default=2,
                    help='number of layers in KAN encoder')
parser.add_argument('--kan_decoder_layers', type=int, default=2,
                    help='number of layers in KAN decoder')
parser.add_argument('--kan_trend_encoder_dim', type=int, nargs='*', default=None,
                    help='hidden dimensions for trend KAN encoder')
parser.add_argument('--kan_season_encoder_dim', type=int, nargs='*', default=None,
                    help='hidden dimensions for season KAN encoder')
parser.add_argument('--kan_trend_decoder_dim', type=int, default=None,
                    help='base hidden dimension for trend KAN decoder')
parser.add_argument('--kan_season_decoder_dim', type=int, default=None,
                    help='base hidden dimension for season KAN decoder')

# KAN Network specific parameters
parser.add_argument('--kan_grid_size', type=int, default=8,
                    help='grid size for KAN B-spline basis functions')
parser.add_argument('--kan_spline_order', type=int, default=3,
                    help='order of B-spline basis functions for KAN')
parser.add_argument('--kan_noise_scale', type=float, default=0.1,
                    help='noise scaling factor for KAN weight initialization')
parser.add_argument('--kan_base_activation', type=str, default='gelu',
                    help='base activation function for KAN, options:[gelu, relu, swish, tanh, none]')
parser.add_argument('--kan_enable_bias', action='store_true', default=True,
                    help='whether to enable bias in KAN layers')
parser.add_argument('--kan_individual', action='store_true', default=False,
                    help='whether to use individual KAN layers for each channel')
parser.add_argument('--kan_hidden_dim', type=int, nargs='*', default=[256],
                help='Hidden dimensions for KAN MLP layers. Can be a single int or space-separated ints (e.g., "256 128 64").')
parser.add_argument('--kan_mlp_layers', type=int, default=2,
                help='Number of KAN MLP layers. Must be at least 1.')
parser.add_argument('--kan_dropout', type=float, default=None,
                    help='dropout rate for KAN layers (if None, use general dropout)')
parser.add_argument('--kan_use_residual', action='store_true', default=True,
                    help='whether to use residual connections in KAN MLP')
parser.add_argument('--kan_use_layer_norm', action='store_true', default=True,
                    help='whether to use layer normalization in KAN layers')
parser.add_argument('--kan_grid_min', type=float, default=-2.0,
                    help='minimum value for learnable KAN grid boundaries')
parser.add_argument('--kan_grid_max', type=float, default=2.0,
                    help='maximum value for learnable KAN grid boundaries')
parser.add_argument('--kan_replace_linear', action='store_true', default=False,
                    help='whether to replace all linear layers with KAN layers in existing models')

# Dynamic Pruning Parameters (DST)
parser.add_argument('--use_dynamic_pruning', action='store_true', default=False,
                    help='whether to enable dynamic pruning (Dynamic Sparse Training) for KAN')
parser.add_argument('--dst_initial_sparsity', type=float, default=0.0,
                    help='initial sparsity for dynamic pruning (0.0 means dense at start)')
parser.add_argument('--dst_target_sparsity', type=float, default=0.5,
                    help='target sparsity for dynamic pruning')
parser.add_argument('--dst_schedule', type=str, default='cosine',
                    help='sparsity schedule for dynamic pruning: [cosine, linear, exponential]')
parser.add_argument('--dst_update_frequency', type=int, default=100,
                    help='frequency (in training steps) to update dynamic pruning mask')
parser.add_argument('--dst_total_steps', type=int, default=10000,
                    help='total training steps for dynamic pruning schedule calculation')

# Patching parameters for xPatch
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--d_state', type=int, default=32, help='d_state')

# Moving Average for xPatch
parser.add_argument('--ma_type', type=str, default='ema', help='reg, ema, dema')
parser.add_argument('--alpha', type=float, default=0.3, help='alpha')
parser.add_argument('--beta', type=float, default=0.3, help='beta')

# Optimization parameters
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU parameters
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')   # cuda or mps
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# De-stationary projector parameters
parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                    help='hidden layer dimensions of projector (List)')
parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

# Metrics (DTW)
parser.add_argument('--use_dtw', type=bool, default=False,
                    help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

# Augmentation parameters
parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
parser.add_argument('--permutation', default=False, action="store_true",
                    help="Equal Length Permutation preset augmentation")
parser.add_argument('--randompermutation', default=False, action="store_true",
                    help="Random Length Permutation preset augmentation")
parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
parser.add_argument('--discdtw', default=False, action="store_true",
                    help="Discrimitive DTW warp preset augmentation")
parser.add_argument('--discsdtw', default=False, action="store_true",
                    help="Discrimitive shapeDTW warp preset augmentation")

# Loss parameters for Sigmoid Loss
parser.add_argument('--conv_weight', type=float, default=0.05, help='Late-stage convergence weight')
parser.add_argument('--decay_k', type=float, default=2.0, help='Attenuation coefficient')


parser.add_argument('--lr_min', type=float, default=1e-6, help='Proportion of stable period')
parser.add_argument('--lr_max', type=float, default=1e-3, help='Transition steepness')
parser.add_argument('--lr_alpha', type=float, default=8.0, help='Transition steepness')
parser.add_argument('--lr_beta', type=float, default=50.0, help='Number of warmup epochs')

# Mutually exclusive group for xPatch
group = parser.add_mutually_exclusive_group()
group.add_argument('--shuffle_half', action='store_true', help='前后半部分交换')
group.add_argument('--shuffle_random', action='store_true', help='完全随机打乱')
args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp_dict = {
            'exp_main': exp_main,
        }
Exp=Exp_dict[args.Exp].Exp_Main


if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(args.model_id,
                                                        args.model,
                                                        args.data,
                                                        args.features,
                                                        args.seq_len,
                                                        args.label_len,
                                                        args.pred_len,
                                                        args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()