import argparse
import copy
import os
import os.path as osp
import time
import warnings
import sys
sys.path.append('/root/autodl-fs/projects/Illumination-Adaptive-Transformer/IAT_high/IAT_mmdetection/')

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger

# python tools/train.py --resume-from /root/autodl-tmp/outputs/IAT/20231229151040_not_IAT/epoch_42.pth

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',default='/root/autodl-fs/projects/Illumination-Adaptive-Transformer/IAT_high/IAT_mmdetection/configs/detr/detr_ours_LOL.py', help='train config file path')
    #parser.add_argument('--config',default='/root/autodl-fs/projects/Illumination-Adaptive-Transformer/IAT_high/IAT_mmdetection/configs/yolo/yolov3_IAT_lol.py', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args

import datetime
def main():
    args = parse_args()
    args.work_dir = "/root/autodl-tmp/outputs/IAT/"+str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))+"_not_IAT"
    print("args", args)
    print("detr not use pre_encoder")
    # args Namespace(cfg_options=None, config='configs/detr/detr_ours_LOL.py', deterministic=False, gpu_ids=None, gpus=None, 
    #                launcher='none', local_rank=0, no_validate=False, options=None, resume_from=None, seed=None, 
    #                work_dir='/root/autodl-tmp/outputs/20231208104502')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    
    print("cfg",cfg)
    
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)


    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()

#cfg Config (path: configs/detr/detr_ours_LOL.py): 
    #     {
    # 	'checkpoint_config': {
    # 		'interval': 1
    # 	},
    # 	'log_config': {
    # 		'interval': 50,
    # 		'hooks': [{
    # 			'type': 'TextLoggerHook'
    # 		}]
    # 	},
    # 	'custom_hooks': [{
    # 		'type': 'NumClassCheckHook'
    # 	}],
    # 	'dist_params': {
    # 		'backend': 'nccl'
    # 	},
    # 	'log_level': 'INFO',
    # 	'load_from': 'https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth',
    # 	'resume_from': None,
    # 	'workflow': [('train', 1)],
    # 	'dataset_type': 'ExdarkDataset',
    # 	'data_root': '/root/autodl-tmp/Exdark/',
    # 	'img_norm_cfg': {
    # 		'mean': [0, 0, 0],
    # 		'std': [255.0, 255.0, 255.0],
    # 		'to_rgb': True
    # 	},
    # 	'train_pipeline': [{
    # 		'type': 'LoadImageFromFile'
    # 	}, {
    # 		'type': 'LoadAnnotations',
    # 		'with_bbox': True
    # 	}, {
    # 		'type': 'RandomFlip',
    # 		'flip_ratio': 0.5
    # 	}, {
    # 		'type': 'AutoAugment',
    # 		'policies': [
    # 			[{
    # 				'type': 'Resize',
    # 				'img_scale': [(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)],
    # 				'multiscale_mode': 'value',
    # 				'keep_ratio': True
    # 			}],
    # 			[{
    # 				'type': 'Resize',
    # 				'img_scale': [(400, 1333), (500, 1333), (600, 1333)],
    # 				'multiscale_mode': 'value',
    # 				'keep_ratio': True
    # 			}, {
    # 				'type': 'RandomCrop',
    # 				'crop_type': 'absolute_range',
    # 				'crop_size': (384, 600),
    # 				'allow_negative_crop': True
    # 			}, {
    # 				'type': 'Resize',
    # 				'img_scale': [(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)],
    # 				'multiscale_mode': 'value',
    # 				'override': True,
    # 				'keep_ratio': True
    # 			}]
    # 		]
    # 	}, {
    # 		'type': 'Normalize',
    # 		'mean': [0, 0, 0],
    # 		'std': [255.0, 255.0, 255.0],
    # 		'to_rgb': True
    # 	}, {
    # 		'type': 'Pad',
    # 		'size_divisor': 1
    # 	}, {
    # 		'type': 'DefaultFormatBundle'
    # 	}, {
    # 		'type': 'Collect',
    # 		'keys': ['img', 'gt_bboxes', 'gt_labels']
    # 	}],
    # 	'test_pipeline': [{
    # 		'type': 'LoadImageFromFile'
    # 	}, {
    # 		'type': 'MultiScaleFlipAug',
    # 		'img_scale': (1333, 800),
    # 		'flip': False,
    # 		'transforms': [{
    # 			'type': 'Resize',
    # 			'keep_ratio': True
    # 		}, {
    # 			'type': 'RandomFlip'
    # 		}, {
    # 			'type': 'Normalize',
    # 			'mean': [0, 0, 0],
    # 			'std': [255.0, 255.0, 255.0],
    # 			'to_rgb': True
    # 		}, {
    # 			'type': 'Pad',
    # 			'size_divisor': 1
    # 		}, {
    # 			'type': 'ImageToTensor',
    # 			'keys': ['img']
    # 		}, {
    # 			'type': 'Collect',
    # 			'keys': ['img']
    # 		}]
    # 	}],
    # 	'data': {
    # 		'samples_per_gpu': 1,
    # 		'workers_per_gpu': 2,
    # 		'train': {
    # 			'type': 'ExdarkDataset',
    # 			'ann_file': '/root/autodl-tmp/Exdark/main/train.txt',
    # 			'img_prefix': '/root/autodl-tmp/Exdark/JPEGImages/IMGS',
    # 			'pipeline': [{
    # 				'type': 'LoadImageFromFile'
    # 			}, {
    # 				'type': 'LoadAnnotations',
    # 				'with_bbox': True
    # 			}, {
    # 				'type': 'RandomFlip',
    # 				'flip_ratio': 0.5
    # 			}, {
    # 				'type': 'AutoAugment',
    # 				'policies': [
    # 					[{
    # 						'type': 'Resize',
    # 						'img_scale': [(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)],
    # 						'multiscale_mode': 'value',
    # 						'keep_ratio': True
    # 					}],
    # 					[{
    # 						'type': 'Resize',
    # 						'img_scale': [(400, 1333), (500, 1333), (600, 1333)],
    # 						'multiscale_mode': 'value',
    # 						'keep_ratio': True
    # 					}, {
    # 						'type': 'RandomCrop',
    # 						'crop_type': 'absolute_range',
    # 						'crop_size': (384, 600),
    # 						'allow_negative_crop': True
    # 					}, {
    # 						'type': 'Resize',
    # 						'img_scale': [(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)],
    # 						'multiscale_mode': 'value',
    # 						'override': True,
    # 						'keep_ratio': True
    # 					}]
    # 				]
    # 			}, {
    # 				'type': 'Normalize',
    # 				'mean': [0, 0, 0],
    # 				'std': [255.0, 255.0, 255.0],
    # 				'to_rgb': True
    # 			}, {
    # 				'type': 'Pad',
    # 				'size_divisor': 1
    # 			}, {
    # 				'type': 'DefaultFormatBundle'
    # 			}, {
    # 				'type': 'Collect',
    # 				'keys': ['img', 'gt_bboxes', 'gt_labels']
    # 			}]
    # 		},
    # 		'val': {
    # 			'type': 'ExdarkDataset',
    # 			'ann_file': '/root/autodl-tmp/Exdark/main/val.txt',
    # 			'img_prefix': '/root/autodl-tmp/Exdark/JPEGImages/IMGS',
    # 			'pipeline': [{
    # 				'type': 'LoadImageFromFile'
    # 			}, {
    # 				'type': 'MultiScaleFlipAug',
    # 				'img_scale': (1333, 800),
    # 				'flip': False,
    # 				'transforms': [{
    # 					'type': 'Resize',
    # 					'keep_ratio': True
    # 				}, {
    # 					'type': 'RandomFlip'
    # 				}, {
    # 					'type': 'Normalize',
    # 					'mean': [0, 0, 0],
    # 					'std': [255.0, 255.0, 255.0],
    # 					'to_rgb': True
    # 				}, {
    # 					'type': 'Pad',
    # 					'size_divisor': 1
    # 				}, {
    # 					'type': 'ImageToTensor',
    # 					'keys': ['img']
    # 				}, {
    # 					'type': 'Collect',
    # 					'keys': ['img']
    # 				}]
    # 			}]
    # 		},
    # 		'test': {
    # 			'type': 'ExdarkDataset',
    # 			'ann_file': '/root/autodl-tmp/Exdark/main/val.txt',
    # 			'img_prefix': '/root/autodl-tmp/Exdark/JPEGImages/IMGS',
    # 			'pipeline': [{
    # 				'type': 'LoadImageFromFile'
    # 			}, {
    # 				'type': 'MultiScaleFlipAug',
    # 				'img_scale': (1333, 800),
    # 				'flip': False,
    # 				'transforms': [{
    # 					'type': 'Resize',
    # 					'keep_ratio': True
    # 				}, {
    # 					'type': 'RandomFlip'
    # 				}, {
    # 					'type': 'Normalize',
    # 					'mean': [0, 0, 0],
    # 					'std': [255.0, 255.0, 255.0],
    # 					'to_rgb': True
    # 				}, {
    # 					'type': 'Pad',
    # 					'size_divisor': 1
    # 				}, {
    # 					'type': 'ImageToTensor',
    # 					'keys': ['img']
    # 				}, {
    # 					'type': 'Collect',
    # 					'keys': ['img']
    # 				}]
    # 			}]
    # 		}
    # 	},
    # 	'model': {
    # 		'type': 'IAT_DETR',
    # 		'pre_encoder': {
    # 			'type': 'IAT',
    # 			'in_dim': 3,
    # 			'with_global': True,
    # 			'init_cfg': {
    # 				'type': 'Pretrained',
    # 				'checkpoint': 'LOL_pretrain.pth'
    # 			}
    # 		},
    # 		'backbone': {
    # 			'type': 'ResNet',
    # 			'depth': 50,
    # 			'num_stages': 4,
    # 			'out_indices': (3, ),
    # 			'frozen_stages': -1,
    # 			'norm_cfg': {
    # 				'type': 'BN',
    # 				'requires_grad': False
    # 			},
    # 			'norm_eval': True,
    # 			'style': 'pytorch',
    # 			'init_cfg': {
    # 				'type': 'Pretrained',
    # 				'checkpoint': 'torchvision://resnet50'
    # 			}
    # 		},
    # 		'bbox_head': {
    # 			'type': 'DETRHead',
    # 			'num_classes': 12,
    # 			'in_channels': 2048,
    # 			'transformer': {
    # 				'type': 'Transformer',
    # 				'encoder': {
    # 					'type': 'DetrTransformerEncoder',
    # 					'num_layers': 6,
    # 					'transformerlayers': {
    # 						'type': 'BaseTransformerLayer',
    # 						'attn_cfgs': [{
    # 							'type': 'MultiheadAttention',
    # 							'embed_dims': 256,
    # 							'num_heads': 8,
    # 							'dropout': 0.1
    # 						}],
    # 						'feedforward_channels': 2048,
    # 						'ffn_dropout': 0.1,
    # 						'operation_order': ('self_attn', 'norm', 'ffn', 'norm')
    # 					}
    # 				},
    # 				'decoder': {
    # 					'type': 'DetrTransformerDecoder',
    # 					'return_intermediate': True,
    # 					'num_layers': 6,
    # 					'transformerlayers': {
    # 						'type': 'DetrTransformerDecoderLayer',
    # 						'attn_cfgs': {
    # 							'type': 'MultiheadAttention',
    # 							'embed_dims': 256,
    # 							'num_heads': 8,
    # 							'dropout': 0.1
    # 						},
    # 						'feedforward_channels': 2048,
    # 						'ffn_dropout': 0.1,
    # 						'operation_order': ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
    # 					}
    # 				}
    # 			},
    # 			'positional_encoding': {
    # 				'type': 'SinePositionalEncoding',
    # 				'num_feats': 128,
    # 				'normalize': True
    # 			},
    # 			'loss_cls': {
    # 				'type': 'CrossEntropyLoss',
    # 				'bg_cls_weight': 0.1,
    # 				'use_sigmoid': False,
    # 				'loss_weight': 1.0,
    # 				'class_weight': 1.0
    # 			},
    # 			'loss_bbox': {
    # 				'type': 'L1Loss',
    # 				'loss_weight': 5.0
    # 			},
    # 			'loss_iou': {
    # 				'type': 'GIoULoss',
    # 				'loss_weight': 2.0
    # 			}
    # 		},
    # 		'train_cfg': {
    # 			'assigner': {
    # 				'type': 'HungarianAssigner',
    # 				'cls_cost': {
    # 					'type': 'ClassificationCost',
    # 					'weight': 1.0
    # 				},
    # 				'reg_cost': {
    # 					'type': 'BBoxL1Cost',
    # 					'weight': 5.0,
    # 					'box_format': 'xywh'
    # 				},
    # 				'iou_cost': {
    # 					'type': 'IoUCost',
    # 					'iou_mode': 'giou',
    # 					'weight': 2.0
    # 				}
    # 			}
    # 		},
    # 		'test_cfg': {
    # 			'max_per_img': 100
    # 		}
    # 	},
    # 	'optimizer': {
    # 		'type': 'AdamW',
    # 		'lr': 0.0001,
    # 		'weight_decay': 0.0001,
    # 		'paramwise_cfg': {
    # 			'custom_keys': {
    # 				'backbone': {
    # 					'lr_mult': 0.1,
    # 					'decay_mult': 1.0
    # 				}
    # 			}
    # 		}
    # 	},
    # 	'optimizer_config': {
    # 		'grad_clip': {
    # 			'max_norm': 0.1,
    # 			'norm_type': 2
    # 		}
    # 	},
    # 	'lr_config': {
    # 		'policy': 'step',
    # 		'step': [14, 20]
    # 	},
    # 	'runner': {
    # 		'type': 'EpochBasedRunner',
    # 		'max_epochs': 21
    # 	},
    # 	'evaluation': {
    # 		'interval': 1,
    # 		'metric': ['mAP']
    # 	},
    # 	'work_dir': '/root/autodl-tmp/outputs/20231208104709',
    # 	'gpu_ids': range(0, 1)
    # }
