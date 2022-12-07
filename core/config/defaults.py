# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os
import torch

from packaging import version
from yacs.config import CfgNode as CN
from torch.nn import init
# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the maximum image side during training will be
# INPUT.MAX_SIZE_TRAIN, while for testing it will be
# INPUT.MAX_SIZE_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.MODEL = CN()
_C.MODEL.STYLE_TRANSFER = CN()
_C.MODEL.STYLE_TRANSFER.N_RESIDUAL = 4
_C.MODEL.SKELETON_AUG = CN()
_C.MODEL.SKELETON_AUG.GRID_H = 8
_C.MODEL.SKELETON_AUG.GRID_W = 8
_C.MODEL.SKELETON_AUG.Z_EN = True
_C.MODEL.SKELETON_AUG.C_COST_EN = True
_C.MODEL.SKELETON_AUG.PRETRAIN_GBA = ""

_C.MODEL.NUM_CLASSES = 306
_C.MODEL.BACKBONE = ''
_C.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False
_C.MODEL.IMAGENET_PRETRAINED_WEIGHTS = ''
# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
_C.MODEL.PRETRAIN_DIR = ""
# TODO取个好点的名字，和SOLVER统一
_C.MODEL.PRETRAIN_ITER = 0


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.IMAGE_HEIGHT = 64
_C.INPUT.IMAGE_WIDTH = 64
_C.INPUT.IMAGE_CHANNELS = 1

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.DIR = "./data/oracle"
_C.DATASETS.NAME = 'oracle'


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000
_C.SOLVER.DECAY_ITER = 20000
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.ADAM_B1 = 0.5
_C.SOLVER.ADAM_B2 = 0.999
_C.SOLVER.CHECKPOINT_PERIOD = 1000
_C.SOLVER.TEST_PERIOD = 100
# loss weight and target°
_C.SOLVER.SIMILAR_TARGET = 0.15  # range 0~1
_C.SOLVER.TPS_DIFF_TARGET = 0.25  # large than 0, value to avoid mode collaspe
_C.SOLVER.AFFINE_DIFF_TARGET = 0.5  # large than 0, value to avoid mode collaspe
_C.SOLVER.LAMBDA_S = 1.  # loss weight for similar
_C.SOLVER.LAMBDA_T = 10.  # loss weight for TPS difference
_C.SOLVER.LAMBDA_A = 10.  # loss weight for Affine difference
_C.SOLVER.LAMBDA_Z = 1.  # loss weight for Z latten reconstrction
_C.SOLVER.LAMBDA_I = 1.  # loss weight for imagew reconstruction
_C.SOLVER.LAMBDA_G_G = 1.  # loss weight for the Generator of sk_au
_C.SOLVER.CLS_THRESHOLD = 0.5
# pretrain model
_C.SOLVER.PRETRAIN_G_Glyphy = ""

#solver of cls
_C.SOLVER.TYPE = 'SGD'
_C.SOLVER.ADAM_BETA1 = 0.5
_C.SOLVER.LR_POLICY = 'step'
_C.SOLVER.STEPS = []
_C.SOLVER.MAX_ITER_Cls = 500000
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.WEIGHT_DECAY = 0.0
_C.SOLVER.STEP_SIZE = 30000
_C.SOLVER.LRS = []
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.GRAD_L2_CLIP = 0.35
_C.SOLVER.BIAS_DOUBLE_LR = False
_C.SOLVER.BIAS_WEIGHT_DECAY = False
_C.SOLVER.WARM_UP_ITERS = 3000
_C.SOLVER.WARM_UP_FACTOR = 1.0 / 3.0
_C.SOLVER.WARM_UP_METHOD = 'linear'
_C.SOLVER.SCALE_MOMENTUM = True
_C.SOLVER.SCALE_MOMENTUM_THRESHOLD = 1.1
_C.SOLVER.LOG_LR_CHANGE_THRESHOLD = 1.1


# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.DATASETS = ()
_C.TRAIN.AUX_NET_EN = False
_C.TRAIN.AUX_DATASETS = ()
# enable Center Loss
_C.TRAIN.CENTER_LOSS_EN = False

# loss weight
_C.TRAIN.CENTER_LOSS_W = 1.0
_C.TRAIN.AUX_LOSS_W = 1.0

_C.TRAIN.IMG_H = 64
_C.TRAIN.IMG_W = 64

_C.TRAIN.CE_WEIGHT = 0.5
_C.TRAIN.CLS_WEIGHT = 0.5

# number of each class's samples after augmentation, enable when values > 0
_C.TRAIN.AUG_TARGET_NUMBER = -1

# keep image aspect while trainning, didn't support True yet
_C.TRAIN.KEEP_ASPECT = False

# Images *per GPU* in the training minibatch
# Total images per minibatch = TRAIN.IMS_PER_BATCH * NUM_GPUS
_C.TRAIN.IMS_PER_BATCH = 64

# Snapshot (model checkpoint) period
# Divide by NUM_GPUS to determine actual period (e.g., 20000/8 => 2500 iters)
# to allow for linear training schedule scaling
_C.TRAIN.SNAPSHOT_ITERS = 3000

# Tensorboard log file path
_C.TRAIN.BOARD_LOG_PATH = './tensorboard'


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.DATASETS = ()
_C.TEST.PRETRAIN_G_Glyphy = ""
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 8
# Number of detections per image
_C.TEST.DETECTIONS_PER_IMG = 100

_C.TEST.IMG_H = 64
_C.TEST.IMG_W = 64
_C.TEST.KEEP_ASPECT = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "./logs"
_C.NUM_GPUS = 1

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")

# ---------------------------------------------------------------------------- #
# Precision options
# ---------------------------------------------------------------------------- #

# Precision of input, allowable: (float32, float16)
_C.DTYPE = "float32"

# Enable verbosity in apex.amp
_C.AMP_VERBOSE = False

# ---------------------------------------------------------------------------- #
# Data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CN()
_C.DATA_LOADER.NUM_THREADS = 8

# ---------------------------------------------------------------------------- #
# CUDA Choice
# ---------------------------------------------------------------------------- #
_C.CUDANUM = CN()
_C.CUDANUM.FIRST = 0
_C.CUDANUM.SECOND = 1
