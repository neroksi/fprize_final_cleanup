import torch
from pathlib import Path
import numpy as np
from transformers import logging
import os

logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"


Discourse2ID = {
    'Claim': 1,
    'Concluding Statement': 2,
    'Counterclaim': 3,
    'Evidence': 4,
    'Lead': 5,
    'Position': 6,
    'Rebuttal': 7,
}

USE_AMP = True

FOLD_COL_NAME = "new_kfold_skf__k_5_v_1_seed_2021"

ID2Discourse = {id_: discourse for discourse, id_ in Discourse2ID.items()}

TRAIN_ROOT = Path("../../../data/train")
TRAIN_CSV_PATH = Path("../../../data/train_rich_v2_amed.csv")

MAIN_METRIC_NAME = "iov_v2_val"
TRUE_SEG_COEF = 0.60

MAXLEN = 512
NUM_INTERVAL = 2

SEED = 321

MODEL_NAME = "roberta-base"


MODEL_ROOT = Path("../../../models")


TRAIN_BATCH_SIZE = 2
TRAIN_NUM_WORKERS = 0

VAL_BATCH_SIZE = 2
VAL_NUM_WORKERS = 0

DEVICE = torch.device("cpu")

CLIP_GRAD_NORM = 2.0

POSITION_DIV_FACTOR = 3

OPTIMIZER_LR = 5e-6
OPTIMIZER_WEIGHT_DECAY = 0.01 #1e-5
SCHEDULER_ETA_MIN = 8e-7

P_MASK_SIZE_LOW = 0.15
P_MASK_SIZE_HIGH = 0.35
P_MASK_FREQ = 0. #0.80 #0.80

P_RANDOM_START = 0. #0.50
P_START_AT_SEQ_BEGINNING = 1.0 # 0.80 # Prob to start at beginning if not random start
MIN_SEQ_LEN = 4096 #512 # All sequences longer than this could be truncated
FORCE_TRUNC_FREQ = 0.50

PYTORCH_CE_IGNORE_INDEX = -100

STRIDE_MAX_LEN_RATIO = 2

ALPHA_NER = 0.20
ALPHA_SEG = 0.80

P_DROPS = None
N_DROPS = 5


## Inference
MINLENGTH = 5
TEST_ROOT = Path("../../../data/test")

Discourse2Weights = {
    'Claim': 0.127,
    'Concluding Statement': 0.075,
    'Counterclaim': 0.182,
    'Evidence': 0.102,
    'Lead': 0.079,
    'Position': 0.102,
    'Rebuttal': 0.259,
}


ID2Weights = {
    1: 0.20,
    2: 0.80, 
}


def to_1_7(x):
    if x == 0:
        return 0
    return 1 + (x - 1)%NUM_PURE_TARGETS

def init_config():
    global NUM_TARGETS, NUM_PURE_TARGETS, CLASS_WEIGHTS, SEG_CLASS_WEIGHTS

    NUM_PURE_TARGETS = len(Discourse2ID)
    NUM_TARGETS = 1 + NUM_PURE_TARGETS * NUM_INTERVAL
        
    CLASS_WEIGHTS = np.zeros(NUM_TARGETS, dtype=np.float32)

    for i in range(1, NUM_TARGETS):
        if i == 0:
            pos = 0
        elif i < 8:
            pos = 1
        elif i < 15:
            pos = 2
        else:
            pos = 3
            
        CLASS_WEIGHTS[i] = ID2Weights[pos] * Discourse2Weights[ID2Discourse[ to_1_7(i) ]]

    CLASS_WEIGHTS[0] = 0.50 * (1 - sum(Discourse2Weights.values()))
    CLASS_WEIGHTS /= CLASS_WEIGHTS.sum()
    CLASS_WEIGHTS *= NUM_TARGETS

    SEG_CLASS_WEIGHTS = 3*np.array([0.35, 0.25, 0.40], dtype=np.float32)


init_config()

assert "NUM_TARGETS" in globals()