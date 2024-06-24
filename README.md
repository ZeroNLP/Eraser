# Eraser
Released code for the paper [Eraser: Jailbreaking defense in large language models via unlearning harmful knowledge](https://arxiv.org/pdf/2404.05880).

## Installation
Install a python environment and install dependency:
```bash
conda env create -f environment.yaml
```

## Download Models
1. Download the [Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and store it in ./models/Llama2-7b-chat-hf/
2. Download the [Wizard-Vicuna-30B-Uncensored-GPTQ](https://huggingface.co/TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ) and store it in ./models/Wizard-Vicuna-30B-Uncensored-GPTQ/

## Generate Data
1. Set your OpenAI key in line 6 of ./generate_data/entity_extraction.py
2. Execute the following commands to generate data:
```bash
cd generate_data
bash generate.sh
```

## Eraser Training
The trained Lora adapter is stored in "./models/Eraser_Llama2_7b_Lora". If you want to retrain, execute the following commands (Please generate the data before training):
```bash
cd train_Eraser
python train.py
```

## Evaluting the defense capability:
AIM attack:
```bash
cd Evaluate_defense_capability
bash run_AIM.sh
```

AutoDAN attack:
Please refer to https://github.com/SheltonLiu-N/AutoDAN 

GCG attack:
Please refer to https://github.com/llm-attacks/llm-attacks

## Evaluting the general capability:
Please refer to https://github.com/EleutherAI/lm-evaluation-harness

## Cite:
```latex
@article{lu2024eraser,
  title={Eraser: Jailbreaking defense in large language models via unlearning harmful knowledge},
  author={Lu, Weikai and Zeng, Ziqian and Wang, Jianwei and Lu, Zhengdong and Chen, Zelin and Zhuang, Huiping and Chen, Cen},
  journal={arXiv preprint arXiv:2404.05880},
  year={2024}
}
```
