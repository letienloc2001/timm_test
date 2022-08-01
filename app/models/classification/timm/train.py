#!/usr/bin/env python3
import argparse
import time
import yaml
import os
import logging
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint,\
    convert_splitbn_model, model_parameters
from timm import utils
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import ApexScaler, NativeScaler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False


torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('train')


def _parse_args(dataset, configuration):
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    config_parser = argparse.ArgumentParser(description='Training Configuration', add_help=True)
    config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                               help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser(description='PyTorch Model Training')
    parser.add_argument('data_dir', metavar='DIR',
                        help='Path/to/dataset')
    ###################################################################################################
    group = parser.add_argument_group('Dataset parameters')
    group.add_argument('--dataset', '-d', metavar='NAME', default='',
                       help='dataset type (default: ImageFolder/ImageTar)')
    group.add_argument('--train-split', metavar='NAME', default='train',
                       help='dataset train split (default: train)')
    group.add_argument('--val-split', metavar='NAME', default='validation',
                       help='dataset validation split (default: validation)')
    group.add_argument('--dataset-download', action='store_true', default=False,
                       help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
    group.add_argument('--class-map', default={'fake': 0, 'real': 1}, type=str, metavar='TXT_FILE',
                       help='Path to class to idx mapping txt file (default: {"fake": 0, "real": 1})')
    ###################################################################################################
    group = parser.add_argument_group('Model parameters')
    group.add_argument('--model', default='mixnet_s', type=str, metavar='MODEL',
                       help='Name of model to train (default: "mixnet_s")')
    group.add_argument('--pretrained', action='store_true', default=False,
                       help='Start with pretrained version of specified network if available (default: False)')
    group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                       help='Initialize model from this checkpoint (default: none)')
    group.add_argument('--resume', default='', type=str, metavar='PATH',
                       help='Resume full model and optimizer state from checkpoint (default: none)')
    group.add_argument('--no-resume-opt', action='store_true', default=False,
                       help='prevent resume of optimizer state when resuming model')
    group.add_argument('--num-classes', type=int, default=None, metavar='N',
                       help='Number of label classes (default: None => model default)')
    group.add_argument('--gp', default=None, type=str, metavar='POOL',
                       help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    group.add_argument('--img-size', type=int, default=None, metavar='N',
                       help='Image patch size (default: None => model default)')
    group.add_argument('--input-size', default=None, nargs=3, type=int,
                       metavar='N', help='Image dimensions d h w (default: None => model default)')
    group.add_argument('--crop-pct', default=0.4, type=float,
                       metavar='N', help='Input image crop percent')
    group.add_argument('--mean', type=float, nargs=3, default=None, metavar='PCT',
                       help='Mean pixel value (default: None => model default)')
    group.add_argument('--std', type=float, nargs=3, default=None, metavar='PCT',
                       help='Standard deviation pixel value (default: None => model default)')
    group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                       help='Image resize interpolation type (default: None => model default)')
    group.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
                       help='Batch size for training (default: 8)')
    group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                       help='Batch size for validation (default: None)')
    group.add_argument('--channels-last', action='store_true', default=False,
                       help='Use channels_last memory layout')
    scripting_group = group.add_mutually_exclusive_group()
    scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true', default=False,
                                 help='torch.jit.script the full model')
    scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                                 help="Enable AOT Autograd support. (It's recommended to use this option with `--fuser nvfuser` together)")
    group.add_argument('--fuser', default='', type=str,
                       help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
    group.add_argument('--grad-checkpointing', action='store_true', default=False,
                       help='Enable gradient checkpointing through model blocks/stages')
    ###################################################################################################
    group = parser.add_argument_group('Optimizer parameters')
    group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                       help='Optimizer (default: "sgd"')
    group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                       help='Optimizer Epsilon (default: None, use opt default)')
    group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                       help='Optimizer Betas (default: None, use opt default)')
    group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                       help='Optimizer momentum (default: 0.9)')
    group.add_argument('--weight-decay', type=float, default=2e-5, metavar='W',
                       help='Weight decay (default: 2e-5)')
    group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                       help='Clip gradient norm (default: None, no clipping)')
    group.add_argument('--clip-mode', type=str, default='norm',
                       help='Gradient clipping mode. One of ("norm", "value", "agc")')
    group.add_argument('--layer-decay', type=float, default=None,
                       help='layer-wise learning rate decay (default: None)')
    ###################################################################################################
    group = parser.add_argument_group('Learning rate schedule parameters')
    group.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                       help='LR scheduler (default: "cosine"')
    group.add_argument('--lr', type=float, default=0.05, metavar='LR',
                       help='Learning rate (default: 0.05)')
    group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                       help='learning rate noise on/off epoch percentages')
    group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                       help='learning rate noise limit percent (default: 0.67)')
    group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                       help='learning rate noise std-dev (default: 1.0)')
    group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                       help='learning rate cycle len multiplier (default: 1.0)')
    group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                       help='amount to decay each learning rate cycle (default: 0.5)')
    group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                       help='learning rate cycle limit, cycles enabled if > 1')
    group.add_argument('--lr-k-decay', type=float, default=1.0,
                       help='learning rate k-decay for cosine/poly (default: 1.0)')
    group.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                       help='warmup learning rate (default: 0.0001)')
    group.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                       help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    group.add_argument('--epochs', type=int, default=30, metavar='N',
                       help='Number of epochs to train (default: 30)')
    group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                       help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                       help='manual epoch number (useful on restarts)')
    group.add_argument('--decay-milestones', default=[30, 60], type=int, nargs='+', metavar="MILESTONES",
                       help='list of decay epoch indices for multistep lr. must be increasing')
    group.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                       help='epoch interval to decay LR')
    group.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                       help='epochs to warmup LR, if scheduler supports')
    group.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                       help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                       help='patience epochs for Plateau LR scheduler (default: 10')
    group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                       help='LR decay rate (default: 0.1)')
    ###################################################################################################
    group = parser.add_argument_group('Augmentation and regularization parameters')
    group.add_argument('--no-aug', action='store_true', default=False,
                       help='Disable all augmentation, override other aug args')
    group.add_argument('--scale', type=float, nargs='+', default=None, metavar='PCT',
                       help='Random resize scale (recommended: 0.08 1.0, default: None)')
    group.add_argument('--ratio', type=float, nargs='+', default=None, metavar='RATIO',
                       help='Random resize aspect ratio (recommended: 0.75 1.33, default: None)')
    group.add_argument('--hflip', type=float, default=0.,
                       help='Horizontal flip training aug probability (default: 0)')
    group.add_argument('--vflip', type=float, default=0.,
                       help='Vertical flip training aug probability (default: 0)')
    group.add_argument('--color-jitter', type=float, default=None, metavar='PCT',
                       help='Color jitter factor (default: None)')
    group.add_argument('--aa', type=str, default=None, metavar='NAME',
                       help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    group.add_argument('--aug-repeats', type=float, default=0,
                       help='Number of augmentation repetitions (distributed training only) (default: 0)')
    group.add_argument('--aug-splits', type=int, default=0,
                       help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    group.add_argument('--jsd-loss', action='store_true', default=False,
                       help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    group.add_argument('--bce-loss', action='store_true', default=False,
                       help='Enable BCE loss w/ Mixup/CutMix use.')
    group.add_argument('--bce-target-thresh', type=float, default=None,
                       help='Threshold for binarizing softened BCE targets (default: None, disabled)')
    group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                       help='Random erase prob (default: 0.)')
    group.add_argument('--remode', type=str, default='pixel',
                       help='Random erase mode (default: "pixel")')
    group.add_argument('--recount', type=int, default=1,
                       help='Random erase count (default: 1)')
    group.add_argument('--resplit', action='store_true', default=False,
                       help='Do not random erase first (clean) augmentation split')
    group.add_argument('--mixup', type=float, default=0.0,
                       help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    group.add_argument('--cutmix', type=float, default=0.0,
                       help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                       help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    group.add_argument('--mixup-prob', type=float, default=1.0,
                       help='Probability of performing mixup or cutmix when either/both is enabled')
    group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                       help='Probability of switching to cutmix when both mixup and cutmix enabled')
    group.add_argument('--mixup-mode', type=str, default='batch',
                       help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                       help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    group.add_argument('--smoothing', type=float, default=None,
                       help='Label smoothing (default: None)')
    group.add_argument('--train-interpolation', type=str, default='bilinear',
                       help='Training interpolation (random, bilinear, bicubic default: "random")')
    group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                       help='Dropout rate (default: 0.)')
    group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                       help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                       help='Drop path rate (default: None)')
    group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                       help='Drop block rate (default: None)')
    group.add_argument('--ten-crop', action='store_true', default=True,
                       help='TenCrop enabled')
    ###################################################################################################
    group = parser.add_argument_group('Batch norm parameters',
                                      'Only works with gen_efficientnet based models currently.')
    group.add_argument('--bn-momentum', type=float, default=None,
                       help='BatchNorm momentum override (if not None)')
    group.add_argument('--bn-eps', type=float, default=None,
                       help='BatchNorm epsilon override (if not None)')
    group.add_argument('--sync-bn', action='store_true', default=False,
                       help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    group.add_argument('--dist-bn', type=str, default='',
                       help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    group.add_argument('--split-bn', action='store_true', default=False,
                       help='Enable separate BN layers per augmentation split.')
    ###################################################################################################
    group = parser.add_argument_group('Model exponential moving average parameters')
    group.add_argument('--model-ema', action='store_true', default=False,
                       help='Enable tracking moving average of model weights')
    group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                       help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    group.add_argument('--model-ema-decay', type=float, default=0.9998,
                       help='decay factor for model weights moving average (default: 0.9998)')
    ###################################################################################################
    group = parser.add_argument_group('Miscellaneous parameters')
    group.add_argument('--seed', type=int, default=42, metavar='S',
                       help='random seed (default: 42)')
    group.add_argument('--worker-seeding', type=str, default='all',
                       help='worker seed mode (default: all)')
    group.add_argument('--log-interval', type=int, default=50, metavar='N',
                       help='how many batches to wait before logging training status')
    group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                       help='how many batches to wait before writing recovery checkpoint')
    group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                       help='number of checkpoints to keep (default: 10)')
    group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                       help='how many training processes to use (default: 4)')
    group.add_argument('--save-images', action='store_true', default=False,
                       help='save images of input bathes every log interval for debugging')
    group.add_argument('--amp', action='store_true', default=False,
                       help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    group.add_argument('--apex-amp', action='store_true', default=False,
                       help='Use NVIDIA Apex AMP mixed precision')
    group.add_argument('--native-amp', action='store_true', default=False,
                       help='Use Native Torch AMP mixed precision')
    group.add_argument('--no-ddp-bb', action='store_true', default=False,
                       help='Force broadcast buffers for native DDP to off.')
    group.add_argument('--pin-mem', action='store_true', default=False,
                       help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    group.add_argument('--no-prefetcher', action='store_true', default=False,
                       help='disable fast prefetcher')
    group.add_argument('--output', default='', type=str, metavar='PATH',
                       help='path to output folder (default: none, current dir)')
    group.add_argument('--experiment', default='', type=str, metavar='NAME',
                       help='name of train experiment, name of sub-folder for output')
    group.add_argument('--eval-metric', default='loss', type=str, metavar='EVAL_METRIC',
                       help='Best metric (default: "loss"')
    group.add_argument('--tta', type=int, default=0, metavar='N',
                       help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
    group.add_argument("--local-rank", default=0, type=int)
    group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                       help='use the multi-epochs-loader to save time at the beginning of every epoch')
    group.add_argument('--log-wandb', action='store_true', default=False,
                       help='log training and validation metrics to wandb')

    # parse .yaml file
    args_config = config_parser.parse_args(['-c', configuration])
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args([dataset])

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def train(model, dataset, configuration):
    utils.setup_default_logging()
    args, args_text = _parse_args(configuration=configuration, dataset=dataset)
    args.prefetcher = not args.no_prefetcher

    # TODO: Set up
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    args.device = 'cuda:0'
    args.world_size = 1
    args.rank = 0  # global rank
    if args.distributed:
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()
        _logger.info('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                        % (args.rank, args.world_size))
    else:
        _logger.info('Training with a single process on 1 GPUs.')
    assert args.rank >= 0

    # NOTE: DON'T USE BY DEFAULT
    if args.rank == 0 and args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning("You've requested to log metrics to wandb but package not found. "
                            "Metrics not being logged to wandb, try `pip install wandb`")

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    if args.amp:
        # `--amp` chooses native amp before apex (APEX ver not actively maintained)
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True

    if args.apex_amp and has_apex:
        use_amp = 'apex'
    elif args.native_amp and has_native_amp:
        use_amp = 'native'
    elif args.apex_amp or args.native_amp:
        _logger.warning("Neither APEX or native Torch AMP is available, using float32. "
                        "Install NVIDA apex or upgrade to PyTorch 1.6")
    # END NOTE

    utils.random_seed(args.seed, args.rank)

    # #
    # model = create_model(
    #     args.model,
    #     pretrained=args.pretrained,
    #     num_classes=args.num_classes,
    #     drop_rate=args.drop,
    #     drop_connect_rate=args.drop_connect,  # DEPRECATED, use drop_path
    #     drop_path_rate=args.drop_path,
    #     drop_block_rate=args.drop_block,
    #     global_pool=args.gp,
    #     bn_momentum=args.bn_momentum,
    #     bn_eps=args.bn_eps,
    #     scriptable=args.torchscript,
    #     checkpoint_path=args.initial_checkpoint)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    # NOTE: DON'T USE BY DEFAULT
    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)
    # END NOTE

    if args.local_rank == 0:
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)

    # NOTE: DON'T USE BY DEFAULT
    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))
    # END NOTE

    # move model to GPU, enable channels last layout if set
    model.cuda()

    # NOTE: DON'T USE BY DEFAULT
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN preferred unless native amp is activated
            model = convert_syncbn_model(model)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.local_rank == 0:
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)
    if args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)
    # END NOTE

    # TODO: CREATE OPTIMMIZER
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # NOTE: DON'T USE BY DEFAULT
    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if args.local_rank == 0:
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = torch.cuda.amp.autocast
        loss_scaler = NativeScaler()
        if args.local_rank == 0:
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if args.local_rank == 0:
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=args.local_rank == 0)

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if args.local_rank == 0:
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if args.local_rank == 0:
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.local_rank], broadcast_buffers=not args.no_ddp_bb)
    # NOTE: EMA model does not need to be wrapped by DDP
    # END NOTE

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)

    # NOTE: DON'T USE BY DEFAULT
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch

    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)
    # END NOTE

    if args.local_rank == 0:
        _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # TODO: CREATE TRAIN AND EVAL DATASETS
    dataset_train = create_dataset(
        args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        repeats=args.epoch_repeats)

    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size)

    # NOTE: DON'T USE BY DEFAULT
    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)
    # END NOTE

    # create data loaders w/ augmentation pipeline
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']

    loader_train = create_loader(
        dataset_train,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
        crop_pct=data_config['crop_pct'],
        is_anti_spoofing=True,
        ten_crop=args.ten_crop
    )

    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        ten_crop=args.ten_crop,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        crop_pct=data_config['crop_pct'],
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
        is_anti_spoofing=True,
    )

    # NOTE: DON'T USE BY DEFAULT
    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    # END NOTE
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.cuda()
    validate_loss_fn = nn.CrossEntropyLoss().cuda()

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if args.rank == 0:
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = utils.CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.checkpoint_hist)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args, ten_crop=args.ten_crop,
                lr_scheduler=lr_scheduler, saver=saver, output_dir=output_dir,
                amp_autocast=amp_autocast, loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn)

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if args.local_rank == 0:
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast)

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
                ema_eval_metrics = validate(
                    model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast,
                    log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if output_dir is not None:
                utils.update_summary(
                    epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                    write_header=best_metric is None, log_wandb=args.log_wandb and has_wandb)

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

    except KeyboardInterrupt:
        pass
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))


def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args, ten_crop=False,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):

        if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
            if args.prefetcher and loader.mixup_enabled:
                loader.mixup_enabled = False
            elif mixup_fn is not None:
                mixup_fn.mixup_enabled = False

        second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        batch_time_m = utils.AverageMeter()
        data_time_m = utils.AverageMeter()
        losses_m = utils.AverageMeter()
        loss = None

        model.train()

        end = time.time()
        last_idx = len(loader) * (10 if args.ten_crop else 1) - 1
        num_updates = epoch * len(loader) * (10 if args.ten_crop else 1)
        
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            data_time_m.update(time.time() - end)
            print('input:', input.shape)
            print('target:', target.shape)
            if args.ten_crop:  # todo: use ten crop augmentation

                # BEFORE
                # input.shape  = batch_size, 10, (3,400,400)
                # target.shape = batch_size
                # input = torch.flatten(input, start_dim=0, end_dim=1)
                # target = torch.Tensor([entry for entry in target for _ in range(10)])
                # print('*input:', input.shape)
                # print('*target:', target.shape)
                # AFTER
                # input.shape  = 10xbatch_size, (3,400,400)
                # target.shape = 10xbatch_size

                for crop_idx in range(10):
                    start_idx = args.batch_size * crop_idx
                    end_idx = start_idx + args.batch_size
                    crop_input = input[start_idx: end_idx:]
                    crop_target = target[start_idx: end_idx:]
                    # CROP
                    # crop_input.shape  = batch_size, (3,400,400)
                    # crop_target.shape = batch_size

                    if not args.prefetcher:
                        crop_input, crop_target = crop_input.cuda(), crop_target.cuda()
                        if mixup_fn is not None:
                            crop_input, crop_target = mixup_fn(crop_input, crop_target)
                    if args.channels_last:
                        crop_input = crop_input.contiguous(memory_format=torch.channels_last)

                    with amp_autocast():
                        output = model(crop_input)
                        loss = loss_fn(output, crop_target)

                    if not args.distributed:
                        losses_m.update(loss.item(), crop_input.size(0))

                    optimizer.zero_grad()

                    # NOTE: DON'T USE BY DEFAULT
                    if loss_scaler is not None:
                        loss_scaler(
                            loss, optimizer,
                            clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                            parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                            create_graph=second_order)
                    # END NOTE:

                    else:
                        loss.backward(create_graph=second_order)

                        # NOTE: DON'T USE BY DEFAULT
                        if args.clip_grad is not None:
                            utils.dispatch_clip_grad(
                                model_parameters(model, exclude_head='agc' in args.clip_mode),
                                value=args.clip_grad, mode=args.clip_mode)
                        # END NOTE:

                        optimizer.step()

                    # NOTE: DON'T USE BY DEFAULT
                    if model_ema is not None:
                        model_ema.update(model)
                    # END NOTE:

                    torch.cuda.synchronize()
                    num_updates += 1

            else:  # todo: no ten crop augmentation
                # input.shape  = batch_size x (3,400,400)
                # target.shape = batch_size
                if not args.prefetcher:
                    input, target = input.cuda(), target.cuda()
                    if mixup_fn is not None:
                        input, target = mixup_fn(input, target)
                if args.channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)

                with amp_autocast():
                    output = model(input)
                    loss = loss_fn(output, target)

                if not args.distributed:
                    losses_m.update(loss.item(), input.size(0))

                optimizer.zero_grad()
                # NOTE: DON'T USE BY DEFAULT
                if loss_scaler is not None:
                    loss_scaler(
                        loss, optimizer,
                        clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                        parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                        create_graph=second_order)
                # END NOTE:
                else:
                    loss.backward(create_graph=second_order)
                    # NOTE: DON'T USE BY DEFAULT
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(model, exclude_head='agc' in args.clip_mode),
                            value=args.clip_grad, mode=args.clip_mode)
                    # END NOTE:
                    optimizer.step()

                # NOTE: DON'T USE BY DEFAULT
                if model_ema is not None:
                    model_ema.update(model)
                # END NOTE:

                torch.cuda.synchronize()
                num_updates += 1

            batch_time_m.update(time.time() - end)
            if last_batch or batch_idx % args.log_interval == 0:
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                if args.distributed:
                    reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                    losses_m.update(reduced_loss.item(), input.size(0))

                if args.local_rank == 0:
                    _logger.info(
                        'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                        '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        'LR: {lr:.3e}  '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                            epoch,
                            batch_idx, len(loader),
                            100. * batch_idx / last_idx,
                            loss=losses_m,
                            batch_time=batch_time_m,
                            rate=input.size(0) * args.world_size / batch_time_m.val,
                            rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                            lr=lr,
                            data_time=data_time_m))

                    if args.save_images and output_dir:
                        torchvision.utils.save_image(
                            input,
                            os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                            padding=0,
                            normalize=True)

            if saver is not None and args.recovery_interval and (
                    last_batch or (batch_idx + 1) % args.recovery_interval == 0):
                saver.save_recovery(epoch, batch_idx=batch_idx)

            if lr_scheduler is not None:
                lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

            end = time.time()
        # end for

        if hasattr(optimizer, 'sync_lookahead'):
            optimizer.sync_lookahead()

        return OrderedDict([('loss', losses_m.avg)])


def validate(model, loader, loss_fn, args, ten_crop=False, amp_autocast=suppress, log_suffix=''):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    loss = None

    model.eval()

    end = time.time()
    last_idx = len(loader) * (10 if args.ten_crop else 1) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if args.ten_crop:  # todo: use ten crop augmentation
                # BEFORE
                # input.shape  = batch_size, 10, (3,400,400)
                # target.shape = batch_size
                input = torch.flatten(input, start_dim=0, end_dim=1)
                target = torch.Tensor([entry for entry in target for _ in range(10)])
                # AFTER
                # input.shape  = 10xbatch_size, (3,400,400)
                # target.shape = 10xbatch_size
                for crop_idx in range(10):
                    start_idx = args.batch_size * crop_idx
                    end_idx = start_idx + args.batch_size
                    crop_input = input[start_idx: end_idx:]
                    crop_target = target[start_idx: end_idx:]
                    # CROP
                    # crop_input.shape  = batch_size, (3,400,400)
                    # crop_target.shape = batch_size
                    if not args.prefetcher:
                        crop_input = crop_input.cuda()
                        crop_target = crop_target.cuda()
                    if args.channels_last:
                        crop_input = crop_input.contiguous(memory_format=torch.channels_last)

                    with amp_autocast():
                        output = model(crop_input)
                    if isinstance(output, (tuple, list)):
                        output = output[0]

                    # NOTE: DON'T USE BY DEFAULT
                    # augmentation reduction
                    reduce_factor = args.tta
                    if reduce_factor > 1:
                        output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                        crop_target = crop_target[0:crop_target.size(0):reduce_factor]
                    # END NOTE

                    loss = loss_fn(output, crop_target)
                    acc1, acc5 = utils.accuracy(output, crop_target, topk=(1, 5))

                    if args.distributed:
                        reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                        acc1 = utils.reduce_tensor(acc1, args.world_size)
                        acc5 = utils.reduce_tensor(acc5, args.world_size)
                    else:
                        reduced_loss = loss.data

                    torch.cuda.synchronize()

                    losses_m.update(reduced_loss.item(), crop_input.size(0))
                    top1_m.update(acc1.item(), output.size(0))
                    top5_m.update(acc5.item(), output.size(0))

            else:  # todo: no ten crop augumentation
                # input.shape  = batch_size x (3,400,400)
                # target.shape = batch_size
                if not args.prefetcher:
                    input = input.cuda()
                    target = target.cuda()
                if args.channels_last:
                    input = input.contiguous(memory_format=torch.channels_last)

                with amp_autocast():
                    output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                # NOTE: DON'T USE BY DEFAULT
                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]
                # END NOTE

                loss = loss_fn(output, target)
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

                if args.distributed:
                    reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                    acc1 = utils.reduce_tensor(acc1, args.world_size)
                    acc5 = utils.reduce_tensor(acc5, args.world_size)
                else:
                    reduced_loss = loss.data

                torch.cuda.synchronize()

                losses_m.update(reduced_loss.item(), input.size(0))
                top1_m.update(acc1.item(), output.size(0))
                top5_m.update(acc5.item(), output.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if args.local_rank == 0 and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    '{0}: [{1:>4d}/{2}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                    'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                        log_name, batch_idx, last_idx, batch_time=batch_time_m,
                        loss=losses_m, top1=top1_m, top5=top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics



from torchvision import transforms as t
from numpy import inf

class Trainer:
    def __init__(self,
                 model = None,
                 batch_size: int = 8,
                 lr: float = 0.001,
                 weight_decay: float = 0.0001,
                 momentum: float = 0.9,
                 cuda_device: str = "cuda:0"):
        """
        Load the model weights and configure images transformer & device

        Args:
            model: timm model object
            batch_size (int): batch size (default: 8)
            lr (float): learning rate (default: 0.001)
            weight_decay (float): weight decay (default: 0.0001)
            momentum (float): momentum (default: 0.9)
            cuda_device (str): CUDA device (default: cuda:0)
        """

        self.device = torch.device(cuda_device if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.num_crops = 10
        self.transform = t.Compose([
            t.Resize(size=(400, 400)),
            t.TenCrop(size=(224, 224)),
            t.Lambda(lambda crops: [t.ToTensor()(crop) for crop in crops]),
            t.Lambda(lambda crops: [t.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(crop) for crop in crops]),
            t.Lambda(lambda crops: torch.stack(crops))
        ])
        self.model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))
        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr = lr,
            weight_decay = weight_decay,
            momentum = momentum
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self,
              data_set: str,
              num_epochs: int,
              best_model_path: str = 'mixnet_s-anti-spoofing.pth'):
        """
        Train model

        Args:
            data_set (str): path/to/dataset
            num_epochs (int): number of epochs to train
            best_model_path (str): path/to/output_checkpoint_file

        Return:
            the best model path
        """
        # CREATE LOADERS
        train_path = os.path.join(data_set, 'train')
        val_path = os.path.join(data_set, 'validation')

        train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=self.transform)
        val_dataset = torchvision.datasets.ImageFolder(root=val_path, transform=self.transform)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=True)


        _logger.info('🚀 START TRAINING ...')

        min_valid_loss = inf
        train_loss_vals = []
        val_loss_vals = []
        train_accu_vals = []
        val_accu_vals = []

        for epoch in range(num_epochs):
            _logger.info(f'\nEpoch {epoch + 1}/{num_epochs}: ')

            # TRAINING Process
            train_loss = 0.0
            correct = 0
            total = 0
            train_epoch_loss = []

            self.model.train()
            for _, (images, labels) in enumerate(train_loader):
                crop_list = images.tolist()
                for crop_idx in range(self.num_crops):
                    cropped_images = torch.Tensor([crop_list[batch_idx][crop_idx] for batch_idx in range(images.size(0))])

                    cropped_images, labels = cropped_images.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(cropped_images)
                    _, predictions = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predictions == labels).sum().item()

                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    train_epoch_loss.append(loss.item())
                    self.optimizer.step()
                    train_loss += loss.item()

            _logger.info(f'- Training Accuracy  : {100 * correct / total:.2f} %, \
                Training Loss  : {train_loss / (len(train_loader) * self.num_crops):.5f}')

            train_loss_vals.append(sum(train_epoch_loss) / len(train_epoch_loss))
            train_accu_vals.append(100 * correct / total)

            # VALIDATION Process
            valid_loss = 0.0
            correct = 0
            total = 0
            val_epoch_loss = []

            self.model.eval()
            for _, (images, labels) in enumerate(val_loader):
                crop_list = images.tolist()
                for crop_idx in range(self.num_crops):
                    cropped_images = torch.Tensor([crop_list[batch_idx][crop_idx] for batch_idx in range(images.size(0))])
                    cropped_images, labels = cropped_images.to(self.device), labels.to(self.device)

                    outputs = self.model(cropped_images)

                    _, predictions = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predictions == labels).sum().item()

                    loss = self.criterion(outputs, labels)
                    val_epoch_loss.append(loss.item())
                    valid_loss += loss.item()

            _logger.info(f'- Validation Accuracy: {100 * correct / total:.2f} %, \
                Validation Loss: {valid_loss / (len(val_loader) * self.num_crops):.5f}')

            val_loss_vals.append(sum(val_epoch_loss) / len(val_epoch_loss))
            val_accu_vals.append(100 * correct / total)

            # TODO: SAVE MODEL CHECKPOINT, EXCEPT THE LAST SOFTMAX LAYER
            if min_valid_loss > valid_loss:
                _logger.info(f'🎯 CHECKPOINT: \
                    Validation Loss ({min_valid_loss / (len(val_loader) * self.num_crops):.5f} \
                        ==> {valid_loss / (len(val_loader) * self.num_crops):.5f})')
                min_valid_loss = valid_loss
                mixnet_s, _ = self.model.children()
                torch.save(mixnet_s.state_dict(), best_model_path)

        return best_model_path