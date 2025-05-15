<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Finetuning LLaMA on commonsense reasoning tasks using MoRAOM

This directory includes the MoRAOM implementation and guidelines for reproducing the results in our paper.

## Setup
1. Install dependencies
```bash
conda create -n mlora_llama python=3.10
conda activate mlora_llama
pip install -r requirements.txt
```

## Datasets
1. Download the complete commonsense datasets from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/tree/main/dataset) and download the commonsense 170k finetuning dataset from [here](https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json), then organize the data as follows
```bash
# Store the complete commonsense datasets
./dataset
# rest of the files
./experiment
./peft
# Finetuning commonsense dataset
./commonsense_170k.json
...
```

## Code Structure

Refer to `./peft/src/peft/tuners/lora_osb.py` for the implementation of MoRAOM.

Refer to `./finetune.py` for finetuning LLaMA using MoRAOM.

Refer to `./commonsense_evaluate.py` for the evaluation of the finetuned model.

## Finetuning and Evaluation

### Finetuning (`./llama_7B_MoRAOM_train_fenbushi_cosb.sh`)
This file contains the code to finetune LLaMA-7B using MoRAOM. 

An example could be:
```
sh llama_7B_MoRAOM_train_fenbushi_cosb.sh
```

### Evaluation 

An example could be:
```
sh llama_7B_MoRAOM_eval_osb_all.sh
```

## Acknowledgement
We greatly appreciate the contributions of two remarkable repositories: [LLM-Adapter](https://github.com/AGI-Edgerunners/LLM-Adapters), [PEFT](https://github.com/huggingface/peft). These projects have significantly benefited our work.

