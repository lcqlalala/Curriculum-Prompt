# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


CUDA_VISIBLE_DEVICES=1 python commonsense_evaluate.py \
    --model LLaMA-7B \
    --adapter LoRA \
    --dataset hellaswag \
    --base_model '/mnt/ckptstorage/user-fs/luankexin/lcq/checkpoints/llama-7b-hf' \
    --batch_size 10 \
    --lora_weights /mnt/ckptstorage/user-fs/luankexin/lcq/DoRA-main/commonsense_reasoning/results_test_1
