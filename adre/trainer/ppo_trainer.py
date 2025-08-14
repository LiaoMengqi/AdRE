import copy
from dataclasses import dataclass
from typing import Optional, Union, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from ..models.adre_qwen2 import AdreQwen2ForCausalLM
from .trainer_utils.roles import MuiltRoleModel
from .trainer_utils.temperature_scheduler import TemperatureScheduler
from accelerate import Accelerator


class PPOTrainer:

    def __init__(self, model: nn.Module, tokenizer, dataloader: DataLoader, args):
        """
        Initialize the PPO Trainer.
        """
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader

        # frozen model parameters
        trainable_para_name_list = ["adapter", "router_proj", "adre_embedding"]
        for name, param in model.named_parameters():
            if not any(
                trainable_name in name for trainable_name in trainable_para_name_list
            ):
                param.requires_grad = False
            else:
                param.requires_grad = True

        # lora+ setup, set higher learning rate for lora_b parameters
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "lora_b" not in n and p.requires_grad
                ],
                "lr": args.lr,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if "lora_b" in n and p.requires_grad
                ],
                "lr": args.lr * args.lora_b_lr_eta,
            },
        ]

        # 创建优化器
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # self.scheduler = get_scheduler(
        #     'cosine_with_min_lr',
        #     optimizer=self.optimizer,
        #     num_warmup_steps=int(0.1 * args.max_step),
        #     num_training_steps=args.max_step,
        #     scheduler_specific_kwargs={'min_lr': 1e-7}
        # )

        self.temperature_scheduler: TemperatureScheduler = TemperatureScheduler(
            self.args.temperature, self.args.final_temperature, self.args.max_step
        )

        self.accelerator = Accelerator()

        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.max_step, eta_min=1e-7
        )

        self.scheduler = self.accelerator.prepare(self.scheduler)


    def train(self):
        self.reference_model = self._clone_model(
            self.actor, self.ref_device
        )  

        loss_list, acc_list, sentences = [], [], []
        step = 0
        for epoch in range(self.args.episodes):
            for data_index, batch in enumerate(self.dataloader):
                if step > self.args.max_step:
                    break
                # print(f"  Step {step + 1}/{len(dataloader)}")
                prompts = batch["prompts"]
                samples_list = self.generate_samples(prompts, self.actor)
                experiences = []
                for i, samples in enumerate(samples_list):
                    samples.case = batch["cases"][i]
                    experiences.append(self.make_experience(samples).to_device("cpu"))
                    sentences.append(experiences[-1].info["sentence"])

                # Perform GRPO updates
                for grpo_iteration in range(self.args.grpo_iterations):
                    loss = self.update_policy(experiences)
                    loss_list.append(loss)

                print()
                if (step) % self.args.save_step == 0:
                    self.actor.model.save_pretrained(
                        f"./checkpoint/{self.args.save_dir}_{(step) // self.args.save_step}/"
                    )
                step += 1
                self.scheduler.step()
                self.temperature_scheduler.step()

        return loss_list, acc_list, sentences
