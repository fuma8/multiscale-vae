#!/bin/bash
#SBATCH --nodelist=ai-gpgpu14
#SBATCH --gres=gpu:1                        # 使用するGPUの数 (The number of GPUs to use)

source ~/.bashrc

export CUDA_VISIBLE_DEVICES=1

echo USED GPUs=$CUDA_VISIBLE_DEVICES

poetry run python scripts/train.py