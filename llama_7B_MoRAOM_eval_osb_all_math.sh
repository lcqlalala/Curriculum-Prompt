# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#!/bin/bash

# 定义数据集数组
datasets=("SVAMP" "AddSub" "MultiArith" "SingleEq" "gsm8k" "AQuA")


# 定义模型和适配器
model="LLaMA3-8B"
adapter="LoRA"
base_model="/mnt/ckptstorage/user-fs/luankexin/lcq/checkpoints/Meta-Llama-3-8B"
batch_size=1

# 获取命令行参数
lora_weights="/mnt/ckptstorage/user-fs//DoRA-main/commonsense_reasoning/results_coling改进_osb_math_sr03_3epoch_8_4_8B_64"
cuda_device=7

# 循环遍历数据集并执行命令
for dataset in "${datasets[@]}"; do
    CUDA_VISIBLE_DEVICES=$cuda_device python mathematical_evaluate.py \
        --model $model \
        --adapter $adapter \
        --dataset $dataset \
        --base_model $base_model \
        --lora_weights $lora_weights | tee -a $lora_weights/$dataset.txt
done
