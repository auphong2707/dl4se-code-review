"""
This file contains all the constants used in the project.
"""

RESULTS_DIR = "./results/"
LOGGING_STEPS = 86
SEED = 42

# [I - CODE SUMMARIZATION]
RESULTS_CS_DIR = RESULTS_DIR + "code-summarization/"
CS_INPUT_MAX_LENGTH = 256
CS_OUTPUT_MAX_LENGTH = 128
CS_NUM_BEAMS = 5

# [1. CODET5-BASE HYPERPARAMETERS]
MODEL_CS_CT5B = "Salesforce/codet5-base"
TOKENIZER_CS_CT5B = "Salesforce/codet5-base"
RESULTS_CS_DIR_CT5B = RESULTS_CS_DIR + "code-t5-base/"

LR_CS_CT5B = 8e-5
TRAIN_BATCH_SIZE_CS_CT5B = 16
EVAL_BATCH_SIZE_CS_CT5B = 16
NUM_TRAIN_EPOCHS_CS_CT5B = 50
WEIGHT_DECAY_CS_CT5B = 0.01
EAS_CS_CT5B = 64
TEC_CS_CT5B = 32

# [2. CODET5-LARGE HYPERPARAMETERS]
MODEL_CS_CT5L = "Salesforce/codet5-large"
TOKENIZER_CS_CT5L = "Salesforce/codet5-large"
RESULTS_CS_DIR_CT5L = RESULTS_CS_DIR + "code-t5-large/"

LR_CS_CT5L = 8e-5
TRAIN_BATCH_SIZE_CS_CT5L = 4
EVAL_BATCH_SIZE_CS_CT5L = 4
NUM_TRAIN_EPOCHS_CS_CT5L = 50
WEIGHT_DECAY_CS_CT5L = 0.01
EAS_CS_CT5L = 64
TEC_CS_CT5L = 32
