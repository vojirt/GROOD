#!/bin/bash
if [ $# -ne 4 ]; then 
    echo "Illegal number of parameters!"
    echo "Usage:"
    echo "  ./run_all_6v4.sh <EXP_OUT_DIR suffix> <config name> <gpu_id> <eval_type>"
    exit 0
fi

EXP_DIR_PREFIX="./_out/experiments/training/5r_6v4/"

if [[ "$2" != "None" ]]; then
    ./eval/scripts/run_5times_6v4.sh $1 cifar10 $2 $3
else
    echo "Skipping computation and running evaluation only!"
fi
CUDA_VISIBLE_DEVICES=$3 python3 eval/eval_6v4.py --exp_dir $EXP_DIR_PREFIX$1/cifar10 --dontknow_prior 0.05 --evaluation_type $4 --latest_checkpoint

if [[ "$2" != "None" ]]; then
    ./eval/scripts/run_5times_6v4.sh $1 mnist $2 $3
else
    echo "Skipping computation and running evaluation only!"
fi
CUDA_VISIBLE_DEVICES=$3 python3 eval/eval_6v4.py --exp_dir $EXP_DIR_PREFIX$1/mnist --dontknow_prior 0.05 --evaluation_type $4 --latest_checkpoint

if [[ "$2" != "None" ]]; then
    ./eval/scripts/run_5times_6v4.sh $1 svhn $2 $3
else
    echo "Skipping computation and running evaluation only!"
fi
CUDA_VISIBLE_DEVICES=$3 python3 eval/eval_6v4.py --exp_dir $EXP_DIR_PREFIX$1/svhn --dontknow_prior 0.05 --evaluation_type $4 --latest_checkpoint

if [[ "$2" != "None" ]]; then
    ./eval/scripts/run_5times_6v4.sh $1 cifar100-10 $2 $3
else
    echo "Skipping computation and running evaluation only!"
fi
CUDA_VISIBLE_DEVICES=$3 python3 eval/eval_6v4.py --exp_dir $EXP_DIR_PREFIX$1/cifar100-10 --dontknow_prior 0.05 --evaluation_type $4 --latest_checkpoint

if [[ "$2" != "None" ]]; then
    ./eval/scripts/run_5times_6v4.sh $1 cifar100-50 $2 $3
else
    echo "Skipping computation and running evaluation only!"
fi
CUDA_VISIBLE_DEVICES=$3 python3 eval/eval_6v4.py --exp_dir $EXP_DIR_PREFIX$1/cifar100-50 --dontknow_prior 0.05 --evaluation_type $4 --latest_checkpoint

if [[ "$2" != "None" ]]; then
    ./eval/scripts/run_5times_6v4.sh $1 tinyimagenet $2 $3
else
    echo "Skipping computation and running evaluation only!"
fi
CUDA_VISIBLE_DEVICES=$3 python3 eval/eval_6v4.py --exp_dir $EXP_DIR_PREFIX$1/tinyimagenet --dontknow_prior 0.05 --evaluation_type $4 --latest_checkpoint
