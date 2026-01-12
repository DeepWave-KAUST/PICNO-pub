#!/bin/bash
#SBATCH --job-name=cno_fd  # 作业名称
#SBATCH --nodes=1                         # 使用的节点数量
#SBATCH --ntasks=1                        # 每个节点的任务数量

#SBATCH --gres=gpu:a100:1                      # 请求 1 个 GPU 资源
#SBATCH --time=02:59:59                   # 最大运行时间
#SBATCH --output=output12.log               # 标准输出文件
#SBATCH --error=error12.log                 # 错误输出文件
#SBATCH --mem=64G
# Shell 脚本运行 Python 命令

python train_cno.py \
--results_path /ibex/project/c2310/12hz_random3_pde16 \
--exp_name 12hz_random3_pde16 \
--gpus 0 \
--in_size 128 \
--train_data_path /ibex/project/c2310/dataset_constant_v_10shots_12Hz_flip_v4/train/ \
--valid_data_path /ibex/project/c2310/dataset_constant_v_10shots_12Hz_flip_v4/validation/ \
--batch_size 32 \
--nn_type cno \
--lr 1e-6 \
--N_layers 5 \
--N_res 4 \
--epochs 800 \
--weight 1 10 1 \
--freq 12 \
--v0 2.80 \