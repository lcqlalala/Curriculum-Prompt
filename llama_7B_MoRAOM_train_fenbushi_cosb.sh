#Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_port=8428 finetune.py \
    --base_model '/mnt/ckptstorage/user-fs/luankexin/lcq/checkpoints/llama-7b-hf' \
    --data_path 'commonsense_170k.json' \
    --output_dir results_coling改进_osb_epoch3_16_8_r64_7B/ \
    --batch_size 16  --micro_batch_size 8 --num_epochs 3 \
    --learning_rate 2e-4 --cutoff_len 256 --val_set_size 120 \
    --eval_step 80 --save_step 80  --adapter_name lora \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r 64 --lora_alpha 128 --osb True --use_gradient_checkpointing

# torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0
# -m torch.distributed.launch --master_port=8679 --nproc_per_node=1
# transformers==4.36.0权重保存出错，使用版本4.36.2

# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port=8679 --nproc_per_node=2 finetune.py \
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 finetune.py \

# '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]'
# '["o_proj", "up_proj", "down_proj"]'
# 2e-4
# 1e-5
# quanxian 2e-4 1200 160 
# '/mnt/ckptstorage/user-fs/luankexin/lcq/checkpoints/llama-7b-hf'
# --osb True
# Meta-Llama-3-8B
