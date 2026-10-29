import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

import random
from dataclasses import dataclass, field
from typing import Optional, List, Union
import numpy as np
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
from utils.Llama_trainer import LlamaTrainer
from utils.Datasets import TrainDataset, EvalDataset
from utils.DataCollator import  DataCollatorForSeq2Seq

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=""
    )

@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data."}
    )
    help_path: str = field(
        default=None,
        metadata={"help": "Path to the training data."}
    )
    algn_path: str = field(
            default=None,
            metadata={"help": "Path to the training data."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    eval_size: Optional[int] = field(default=20)
    training_batch_size: int = field(default=128)
    evaluation_batch_size: int = field(default=128)
    micro_batch_size: int = field(default=1)
    model_max_length: int = field(default=2048)
    pad_to_multiple_of: int = field(default=8)
    r: Optional[int] = field(default=8)
    lora_alpha: Optional[int] = field(default=16)
    target_modules: Optional[List[str]] = field(default=None)
    lora_dropout: Optional[float] = field(default=0.05)
    bias: Optional[str] = field(default="none")
    task_type: Optional[Union[str, TaskType]] = field(default="CAUSAL_LM")
    do_eval: bool = field(default=False)
    harmful_threshold: Optional[float] = field(default=1.5)

def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train():
    # Parsing configuration file
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_json_file("./config/train_agrs.json")

    # Fixed random seed
    set_seed(training_args.seed)

    # Calculate the cumulative gradient steps and batch size for each device
    training_args.gradient_accumulation_steps = (training_args.training_batch_size // training_args.micro_batch_size)
    training_args.per_device_train_batch_size = training_args.micro_batch_size
    training_args.per_device_eval_batch_size = training_args.micro_batch_size

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        use_fast=False,
        cache_dir=training_args.cache_dir,
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = (0)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map = "auto",
        load_in_8bit = True
        )

    model = prepare_model_for_int8_training(model)

    # Configure LoRA hyperparameters
    lora_config = LoraConfig(
        r=training_args.r,
        lora_alpha=training_args.lora_alpha,
        target_modules=training_args.target_modules,
        lora_dropout=training_args.lora_dropout,
        bias=training_args.bias,
        task_type=training_args.task_type
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=training_args.pad_to_multiple_of,
        return_tensors="pt",
        padding=True
    )

    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    # Prepare the datasets
    train_dataset = TrainDataset(
        data_args.data_path, data_args.help_path, data_args.algn_path, tokenizer, training_args.model_max_length
    )
    eval_dataset = EvalDataset(
        data_args.data_path, data_args.help_path, data_args.algn_path, tokenizer, training_args.model_max_length
    )

    trainer = LlamaTrainer(
        original_path=model_args.model_name_or_path,
        harmful_threshold = training_args.harmful_threshold,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=data_collator
    )
    model.config.use_cache = False

    trainer.save_model(training_args.output_dir)

    trainer.train()

    tokenizer.save_pretrained(training_args.output_dir)
    trainer.save_model(training_args.output_dir)

    with open(os.path.join(training_args.output_dir,'log.txt'), 'w') as f:
        for elem in trainer.state.log_history:
            f.write(str(elem))
            f.write('\n')

if __name__ == "__main__":
    train()
