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
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

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
        help="batch size for each forward pass and backward pass",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--use_kl_loss", action="store_true", default=False)
    parser.add_argument("--use_kl_estimator_k3", action="store_true", default=False)
    parser.add_argument("--init_kl_coef", type=float, default=1e-4)
    parser.add_argument("--reward_std", action="store_true", default=False)
    parser.add_argument("--entropy_coeff", type=float, default=1e-3)
    parser.add_argument("--token_level_loss", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--log_prob_batch_size", type=int, default=4)

    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=5)

    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--save_step", type=int, default=20)
    parser.add_argument("--save_dir", type=str, default="./checkpoint/")

    # adapter config
    parser.add_argument("--use_adapter", action="store_true", default=False)
    parser.add_argument("--adapter_layers", type=int, default=1)
    parser.add_argument("--adapter_rank", type=int, default=1)

    # rollout parameters
    parser.add_argument(
        "--rollout_n", type=int, default=8, help="number of samples for each prompt"
    )
    parser.add_argument("--max_len", type=int, default=8196)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument(
        "--use_temperature_scheduler", action="store_true", default=False
    )
    

    args = parser.parse_args()

    main(args)
