#!/bin/bash
if [ $# -ne 4 ]; then 
    echo "Illegal number of parameters!"
    echo "Usage:"
    echo "  ./run_shift_all.sh <EXP_OUT_DIR suffix> <config name> <gpu_id> <eval_type>"
    exit 0
fi

EXP_OUT_DIR="./_out/experiments/shift/$1/"
CONFIG="./config/$2"
GPU=$3 
EVAL_TYPE=$4
echo "EXP_DIR: $EXP_OUT_DIR, CONFIG: $CONFIG, GPU: $GPU, EVAL_TYPE: $EVAL_TYPE"


# eval functions
eval_A () {
    CUDA_VISIBLE_DEVICES=$GPU python3 eval/eval_vs_dataset.py --exp_dir $1 --dontknow_prior 0.05 --evaluation_type $EVAL_TYPE --latest_checkpoint DATASET.TEST $2 DATASET.OOD_SELECTED_LABELS -1,0,172
} 
eval_B () {
    CUDA_VISIBLE_DEVICES=$GPU python3 eval/eval_vs_dataset.py --exp_dir $1 --dontknow_prior 0.05 --evaluation_type $EVAL_TYPE --latest_checkpoint DATASET.TEST $2 DATASET.OOD_SELECTED_LABELS -1,173,344
} 

# Training on Real-A, Clipart-A, Quickdraw-A
DATASET="realA"

if [[ "$2" != "None" ]]; then
    CUDA_VISIBLE_DEVICES=$GPU python3 train.py --config $CONFIG EXPERIMENT.OUT_DIR $EXP_OUT_DIR EXPERIMENT.NAME $DATASET DATASET.TRAIN real DATASET.VAL real DATASET.TEST real DATASET.SELECTED_LABELS -1,0,172 DATASET.OOD_SELECTED_LABELS -1,173,344 MODEL.NUM_CLASSES 173
else
    echo "Skipping training on $DATASET and running evaluation only!"
fi

# realB
eval_B $EXP_OUT_DIR$DATASET real 
# clipartA
eval_A $EXP_OUT_DIR$DATASET clipart 
# clipartB
eval_B $EXP_OUT_DIR$DATASET clipart 
# quickdrawA
eval_A $EXP_OUT_DIR$DATASET quickdraw 
# quickdrawB
eval_B $EXP_OUT_DIR$DATASET quickdraw 

exit 0

DATASET="clipartA"
if [[ "$2" != "None" ]]; then
    CUDA_VISIBLE_DEVICES=$GPU python3 train.py --config $CONFIG EXPERIMENT.OUT_DIR $EXP_OUT_DIR EXPERIMENT.NAME $DATASET DATASET.TRAIN clipart DATASET.VAL clipart DATASET.TEST clipart DATASET.SELECTED_LABELS -1,0,172 DATASET.OOD_SELECTED_LABELS -1,173,344 MODEL.NUM_CLASSES 173
else
    echo "Skipping training on $DATASET and running evaluation only!"
fi

# clipartB
eval_B $EXP_OUT_DIR$DATASET clipart 
# realA
eval_A $EXP_OUT_DIR$DATASET real 
# realB
eval_B $EXP_OUT_DIR$DATASET real 
# quickdrawA
eval_A $EXP_OUT_DIR$DATASET quickdraw 
# quickdrawB
eval_B $EXP_OUT_DIR$DATASET quickdraw 

DATASET="quickdrawA"
if [[ "$2" != "None" ]]; then
    CUDA_VISIBLE_DEVICES=$GPU python3 train.py --config $CONFIG EXPERIMENT.OUT_DIR $EXP_OUT_DIR EXPERIMENT.NAME $DATASET DATASET.TRAIN quickdraw DATASET.VAL quickdraw DATASET.TEST quickdraw DATASET.SELECTED_LABELS -1,0,172 DATASET.OOD_SELECTED_LABELS -1,173,344 MODEL.NUM_CLASSES 173
else
    echo "Skipping training on $DATASET and running evaluation only!"
fi

# quickdrawB
eval_B $EXP_OUT_DIR$DATASET quickdraw 
# realA
eval_A $EXP_OUT_DIR$DATASET real 
# realB
eval_B $EXP_OUT_DIR$DATASET real 
# clipartA
eval_A $EXP_OUT_DIR$DATASET clipart 
# clipartB
eval_B $EXP_OUT_DIR$DATASET clipart 


