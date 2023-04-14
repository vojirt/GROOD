#!/bin/bash
if [ $# -ne 3 ]; then 
    echo "Illegal number of parameters!"
    echo "Usage:"
    echo "  ./run_eval_ood.sh <EXP_DIR> <gpu_id> <eval_type>"
    exit 0
fi

EXP_DIR=$1
GPU=$2 
EVAL_TYPE=$3
echo "EXP_DIR: $EXP_DIR, GPU: $GPU, EVAL_TYPE: $EVAL_TYPE"

datasets=("svhn" "lsun" "texture" "places365" "cifar100" "inaturalist" "tinyimagenet" "mnist")
for test in ${datasets[@]}; do
    CUDA_VISIBLE_DEVICES=$GPU python3 eval/eval_vs_dataset.py --exp_dir $EXP_DIR --dontknow_prior 0.05 --evaluation_type $EVAL_TYPE --latest_checkpoint DATASET.TEST $test INPUT.BATCH_SIZE 128 
done



