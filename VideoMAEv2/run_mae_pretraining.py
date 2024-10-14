# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import datetime
import json
import os
import random
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from packaging import version
from timm.models import create_model
from torchvision import transforms

# NOTE: Do not comment `import models`, it is used to register models
import models  # noqa: F401
import utils
#from dataset import build_pretraining_dataset
from engine_for_pretraining import train_one_epoch
from optim_factory import create_optimizer
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_pretrain_samples_collate
from data.dataloader import VideoFrameDataset


def get_args():
    parser = argparse.ArgumentParser(
        'VideoMAE v2 pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)

    # Model parameters
    parser.add_argument(
        '--model',
        default='pretrain_videomae_base_patch16_224',
        type=str,
        metavar='MODEL',
        help='Name of model to train')
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument(
        '--with_checkpoint', action='store_true', default=False)

    parser.add_argument(
        '--decoder_depth', default=4, type=int, help='depth of decoder')

    parser.add_argument(
        '--mask_type',
        default='tube',
        choices=['random', 'tube'],
        type=str,
        help='encoder masked strategy')
    parser.add_argument(
        '--decoder_mask_type',
        default='run_cell',
        choices=['random', 'run_cell'],
        type=str,
        help='decoder masked strategy')

    parser.add_argument(
        '--mask_ratio', default=0.9, type=float, help='mask ratio of encoder')
    parser.add_argument(
        '--decoder_mask_ratio',
        default=0.0,
        type=float,
        help='mask ratio of decoder')

    parser.add_argument(
        '--input_size',
        default=224,
        type=int,
        help='images input size for backbone')

    parser.add_argument(
        '--drop_path',
        type=float,
        default=0.0,
        metavar='PCT',
        help='Drop path rate (default: 0.1)')

    parser.add_argument(
        '--normlize_target',
        default=True,
        type=bool,
        help='normalized the target patch pixels')

    # Optimizer parameters
    parser.add_argument(
        '--opt',
        default='adamw',
        type=str,
        metavar='OPTIMIZER',
        help='Optimizer (default: "adamw"')
    parser.add_argument(
        '--opt_eps',
        default=1e-8,
        type=float,
        metavar='EPSILON',
        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument(
        '--opt_betas',
        default=None,
        type=float,
        nargs='+',
        metavar='BETA',
        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument(
        '--clip_grad',
        type=float,
        default=None,
        metavar='NORM',
        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        metavar='M',
        help='SGD momentum (default: 0.9)')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.05,
        help='weight decay (default: 0.05)')
    parser.add_argument(
        '--weight_decay_end',
        type=float,
        default=None,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)"""
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1.5e-4,
        metavar='LR',
        help='learning rate (default: 1.5e-4)')
    parser.add_argument(
        '--warmup_lr',
        type=float,
        default=1e-6,
        metavar='LR',
        help='warmup learning rate (default: 1e-6)')
    parser.add_argument(
        '--min_lr',
        type=float,
        default=1e-5,
        metavar='LR',
        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument(
        '--warmup_epochs',
        type=int,
        default=40,
        metavar='N',
        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=-1,
        metavar='N',
        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument(
        '--color_jitter',
        type=float,
        default=0.0,
        metavar='PCT',
        help='Color jitter factor (default: 0.4)')
    parser.add_argument(
        '--train_interpolation',
        type=str,
        default='bicubic',
        choices=['random', 'bilinear', 'bicubic'],
        help='Training interpolation')

    # * Finetuning params
    parser.add_argument(
        '--finetune', default='', help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument(
        '--data_path',
        default='/your/data/annotation/path',
        type=str,
        help='dataset path')
    parser.add_argument(
        '--data_root', default='', type=str, help='dataset path root')
    parser.add_argument(
        '--fname_tmpl',
        default='img_{:05}.jpg',
        type=str,
        help='filename_tmpl for rawframe data')
    parser.add_argument(
        '--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--num_sample', type=int, default=1)
    parser.add_argument(
        '--output_dir',
        default='',
        help='path where to save, empty for no saving')
    parser.add_argument(
        '--log_dir', default=None, help='path where to tensorboard log')
    parser.add_argument(
        '--device',
        default='cuda',
        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument(
        '--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument(
        '--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument(
        '--pin_mem',
        action='store_true',
        help=
        'Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument(
        '--no_pin_mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        '--world_size',
        default=1,
        type=int,
        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument(
        '--dist_url',
        default='env://',
        help='url used to set up distributed training')

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        all_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        decoder_depth=args.decoder_depth,
        with_cp=args.with_checkpoint)

    if version.parse(torch.__version__) > version.parse('1.13.1'):
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)

    return model


def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // args.tubelet_size,
                        args.input_size // patch_size[0],
                        args.input_size // patch_size[1])
    args.patch_size = patch_size

    # get dataset
    #dataset_train = build_pretraining_dataset(args)

    """
    Dataset loading
    """
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../depth_anything_v2'))

    # Import the Depth Anything  V2 model from the specified module.
    from depth_anything_v2.dpt import DepthAnythingV2

    # Load configuration settings from a YAML file (assuming you have a 'config.yaml' file).
    import yaml
    with open('../config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Configure the device to be used for computation.
    # Use GPU ('cuda') if available; otherwise, fall back to CPU ('cpu').
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define model configurations for the Depth Anything V2 model.
    # The 'vitl' configuration specifies the model architecture and parameters.
    model_configs = {
        'vitl': {
            'encoder': 'vitl',               # Use the 'vitl' encoder architecture.
            'features': 256,                 # Number of feature maps.
            'out_channels': [256, 512, 1024, 1024]  # Output channels at different layers.
        }
    }

    # Initialize the Depth Anything V2 model with the specified configuration.
    depth_model = DepthAnythingV2(**model_configs['vitl'])

    # Suppress future warnings during model loading.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        # Load the pre-trained model weights from the checkpoint specified in the configuration.
        depth_model.load_state_dict(torch.load(config['data']['depth_model_checkpoint'], map_location=DEVICE))

    # Move the model to the configured device (GPU or CPU) and set it to evaluation mode.
    depth_model = depth_model.to(DEVICE).eval()

    # Set the random seed for reproducibility.
    random.seed(24)
    torch.manual_seed(24)

    # Define the transformation pipeline to apply to each frame.
    transform = transforms.Compose([
        transforms.ToPILImage(),         # Convert the frame to a PIL Image.
        transforms.Resize((224, 224)),   # Resize the image to 224x224 pixels.
        transforms.ToTensor(),           # Convert the image to a PyTorch tensor with shape (3, 224, 224).
    ])

    # Define the number of frames to sample per sequence.
    NUM_FRAMES = 32  # T = 32

    # Define the frame interval (e.g., take every nth frame).
    FRAME_INTERVAL = 4  # Take one frame every 4 frames.

    # Specify the path to the folder containing video files.
    video_folder = config['data']['finevideo_path'] + '/sports_videos'

    # Check if the specified video folder exists.
    if not os.path.exists(video_folder):
        video_folder = '/home/ndelafuente/VD-MAE/sports_videos'
        if not os.path.exists(video_folder):
            # If the folder does not exist, raise a FileNotFoundError.
            raise FileNotFoundError(f"The specified video folder does not exist: {video_folder}")

    dataset = VideoFrameDataset(
        video_folder=video_folder,       # Path to the video folder.
        transform=transform,             # Transformation to apply to each frame.
        depth_model=depth_model,         # Depth estimation model.
        num_frames=NUM_FRAMES,           # Number of frames per sequence.
        frame_interval=FRAME_INTERVAL    # Frame interval for sampling.
    )

    # Split the dataset into training and validation sets.
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [train_size, val_size])

    """
    End of dataset loading
    """

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rank = global_rank
    total_batch_size = args.batch_size * num_tasks

    num_training_steps_per_epoch = len(dataset_train) // total_batch_size

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True)
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if args.num_sample > 1:
        collate_func = partial(multiple_pretrain_samples_collate, fold=False)
    else:
        collate_func = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
        worker_init_fn=utils.seed_worker,
        persistent_workers=True)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in ['model', 'module']:
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint

        utils.load_state_dict(model, checkpoint_model)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))

    # scale the lr
    args.lr = args.lr * total_batch_size / 256
    args.min_lr = args.min_lr * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256

    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" %
          (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(args.weight_decay,
                                                args.weight_decay_end,
                                                args.epochs,
                                                num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" %
          (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler)
    torch.cuda.empty_cache()
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            patch_size=patch_size[0],
            normlize_target=args.normlize_target)
        if args.output_dir:
            _epoch = epoch + 1
            if _epoch % args.save_ckpt_freq == 0 or _epoch == args.epochs:
                utils.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch)

        log_stats = {
            **{f'train_{k}': v
               for k, v in train_stats.items()}, 'epoch': epoch,
            'n_parameters': n_parameters
        }

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                    os.path.join(args.output_dir, "log.txt"),
                    mode="a",
                    encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)