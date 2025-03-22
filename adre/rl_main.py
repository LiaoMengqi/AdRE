import json
import os
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset

from .trainer.ppo_trainer import PPOTrainer
from .models.adre_qwen2 import AdreQwen2ForCausalLM
from .models.adre_llama import AdreLLaMAForCausalLM
from .datasets.adre_rl_datasets import PromptDataset, adre_collate_fn
from .utils.utils import set_seed


def main(args):
    set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model)

    if config.model_type == "llama":
        model = AdreLLaMAForCausalLM.from_pretrained(args.model)
    elif config.model_type == "qwen2":
        model = AdreQwen2ForCausalLM.from_pretrained(args.model)
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"

    dataset = load_dataset(args.dataset, split="train")
    dataset = PromptDataset(dataset, tokenizer, max_prompt_length=args.max_len)

    trainer=PPOTrainer(model,tokenizer,dataset,args)
    trainer.train()

    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # general
    parser.add_argument("--model", type=str, default="./checkpoint/")
    parser.add_argument("--dataset", type=str, default="./dataset/train_cases.json")
    parser.add_argument("--seed", type=int, default=0)

    # training
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="number of episodes to train on the dataset",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="epochs train on experience"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="number of prompts to rollout per batch",
    )
    parser.add_argument(
        "--mini_batch_size",
        type=int,
        default=5,
        help="batch size of experience to update policy",
    )
    parser.add_argument(
        "--micro_batch_size_per_gpu",
        type=int,
        default=5,
        help="batch size to update policy",
    )
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=5)

    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--save_step", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="./checkpoint/")

    # rollout parameters
    parser.add_argument(
        "--rollout_n", type=int, default=10, help="number of samples for each prompt"
    )
    parser.add_argument("--max_len", type=int, default=8196)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument(
        "--use_temperature_scheduler", action="store_true", default=False
    )
    parser.add_argument("--final_temperature", type=float, default=0.8)
    

    args = parser.parse_args()

    main(args)
