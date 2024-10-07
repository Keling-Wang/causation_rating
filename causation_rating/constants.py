import torch
from transformers import set_seed

__all__ = ['SEED', 'DEVICE', 'EPOCH', 'BATCH_SIZE', 'DIST_MATRIX', 
           'LR_SEARCH', 'WEIGHT_DECAY_SEARCH', 'WARMUP_RATIO_SEARCH', 'LR_SCHEDULER_POWER_SEARCH', 'OLL_POWER_SEARCH', 
           'MODEL_PATH_AUG', 'MODEL_PATH_PRE', 'MODEL_PATH_FINALUSE', 
           'DATASET_PATH', 'IMPORTANT_WORD_PATH', 'OUTPUT_PATH', 'TEST_DATA_NAME']

SEED = 114514
set_seed(SEED)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 8
BATCH_SIZE = 128

DIST_MATRIX = torch.tensor([[0, 2, 3.5, 4.5, 5.5],
                            [2, 0, 1.5, 2.5, 3.5],
                            [3.5, 1.5, 0, 1, 2],
                            [4.5, 2.5, 1, 0, 1],
                            [5.5, 3.5, 2, 1, 0]])

LR_SEARCH = [5e-6, 8e-5]
WEIGHT_DECAY_SEARCH = [0.035, 0.15]
WARMUP_RATIO_SEARCH = [0.30, 0.50]
LR_SCHEDULER_POWER_SEARCH = [1.0, 3.0]
OLL_POWER_SEARCH = [2.0, 2.25, 2.5, 2.75, 3.0]

MODEL_PATH_AUG = 'dmis-lab/biobert-base-cased-v1.2'
MODEL_PATH_PRE = 'kelingwang/bert-causation-rating-pubmed'
MODEL_PATH_FINALUSE = 'kelingwang/bert-causation-rating-dr1'

DATASET_PATH = 'https://huggingface.co/datasets/kelingwang/causation_strength_rating/resolve/main/rating_dr1.csv'
IMPORTANT_WORD_PATH = 'https://huggingface.co/datasets/kelingwang/causation_strength_rating/resolve/main/linkingwords_complete.csv'

OUTPUT_PATH = './'

TEST_DATA_NAME = 'test_dr1.csv'

def _set_config(**kwargs):
    global SEED, DEVICE, EPOCH, BATCH_SIZE, DIST_MATRIX, LR_SEARCH, WEIGHT_DECAY_SEARCH, WARMUP_RATIO_SEARCH, LR_SCHEDULER_POWER_SEARCH, OLL_POWER_SEARCH, MODEL_PATH_AUG, MODEL_PATH_PRE, MODEL_PATH_FINALUSE, DATASET_PATH, IMPORTANT_WORD_PATH, OUTPUT_PATH, TEST_DATA_NAME
      # Include all constants you want to be configurable

    for key, value in kwargs.items():
        if key == 'SEED':
            SEED = value
            set_seed(SEED)
        elif key == 'DEVICE':
            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #os.environ["CUDA_VISIBLE_DEVICES"] = "1" if torch.cuda.is_available() else ""
        elif key == 'EPOCH':
            EPOCH = value
        elif key == 'BATCH_SIZE':
            BATCH_SIZE = value
        elif key == 'DIST_MATRIX':
            DIST_MATRIX = value
        elif key == 'LR_SEARCH':
            LR_SEARCH = value
        elif key == 'WEIGHT_DECAY_SEARCH':
            WEIGHT_DECAY_SEARCH = value
        elif key == 'WARMUP_RATIO_SEARCH':
            WARMUP_RATIO_SEARCH = value
        elif key == 'LR_SCHEDULER_POWER_SEARCH':
            LR_SCHEDULER_POWER_SEARCH = value
        elif key == 'OLL_POWER_SEARCH':
            OLL_POWER_SEARCH = value
        elif key == 'MODEL_PATH_AUG':
            MODEL_PATH_AUG = value
        elif key == 'MODEL_PATH_PRE':
            MODEL_PATH_PRE = value
        elif key == 'MODEL_PATH_FINALUSE':
            MODEL_PATH_FINALUSE = value
        elif key == 'DATASET_PATH':
            DATASET_PATH = value
        elif key == 'IMPORTANT_WORD_PATH':
            IMPORTANT_WORD_PATH = value
        elif key == 'OUTPUT_PATH':
            OUTPUT_PATH = value
        elif key == 'TEST_DATA_NAME':
            TEST_DATA_NAME = value
        else:
            raise UserWarning(f"Unknown config key: {key}")