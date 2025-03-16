import glob
import os
import random

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments, Trainer, AutoConfig, )

from adre.models.adre_qwen2 import AdreQwen2ForCausalLM


# 将预处理函数移到全局作用域
def preprocess_function(examples, tokenizer, num_experts):
    batch_size = len(examples["prompt"])

    input_texts = tokenizer.apply_chat_template(examples['prompt'], tokenize=False,
                                                add_generation_prompt=True)

    outputs = []
    for i in range(batch_size):
        outputs.append(examples['deepseek_reasoning'][i] + '\n</think>\n\n' + examples['deepseek_solution'][
            i] + '<｜end▁of▁sentence｜>')

    tokenized_inputs = tokenizer(
        input_texts,
        truncation=True,
        max_length=8192,  # 8k
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )
    input_lengths = [len(tokenized_input_ids) for tokenized_input_ids in tokenized_inputs["input_ids"]]
    max_length = max(input_lengths)

    tokenized_outputs = tokenizer(
        outputs,
        truncation=True,
        max_length=8192 - max_length,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )
    output_lengths = [len(tokenized_output_ids) for tokenized_output_ids in tokenized_outputs["input_ids"]]

    labels = []
    input_ids = []
    # output_mask = []
    gate_values = []
    for i in range(batch_size):
        label = [-100] * (input_lengths[i]) + tokenized_outputs["input_ids"][i]
        labels.append(label)
        input_ids.append(tokenized_inputs["input_ids"][i] + tokenized_outputs["input_ids"][i])
        # output_mask.append([0] * (input_lengths[i]) + [1] * output_lengths[i])
        gate_value = [0] * num_experts
        gate_value[max(0, int(examples['rank'][i] * num_experts - 1e-5))] = 1
        gate_values.append(gate_value)
    model_inputs = {'input_ids': input_ids,
                    'labels': labels,
                    'input_len': input_lengths,
                    'target_length': output_lengths,
                    # 'output_mask': output_mask,
                    'gate_values': gate_values}
    return model_inputs


# 3. 准备数据集
def load_and_process_data(data_path, tokenizer, base_length):
    file_paths = glob.glob(f"{data_path}/*.arrow")

    # 加载并合并所有文件的数据集
    dataset = load_dataset("arrow", data_files=file_paths, split="train")

    processed_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer, base_length),
        batched=True,
        remove_columns=dataset.column_names
    )

    return processed_dataset


class AdreDataCollator(DataCollatorForSeq2Seq):

    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors=None)
        max_length = max(len(feature["input_ids"]) for feature in features)
        padded_position_ids = []
        output_mask = []
        for feature in features:
            orig_position_ids = list(range(feature["input_len"] + feature["target_length"]))

            padding_length = max_length - len(orig_position_ids)
            if self.tokenizer.padding_side == "right":
                padded_ids = orig_position_ids + [0] * padding_length
            else:
                padded_ids = [0] * padding_length + orig_position_ids
            padded_position_ids.append(padded_ids)
            output_mask.append([0] * (max_length - feature['target_length']) + [1] * feature['target_length'])

        batch["position_ids"] = torch.tensor(padded_position_ids, dtype=torch.long, device=batch["input_ids"].device)
        batch["output_mask"] = torch.tensor(output_mask, dtype=torch.long, device=batch["input_ids"].device)
        return batch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # general args
    parser.add_argument("--model", type=str,
                        default="/data/MaoXiaowei/models/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1___5B/",
                        help="model name or path")
    parser.add_argument("--dataset", type=str, default="/data/MaoXiaowei/models/adre/dataset/mix_filtered_45k/",
                        help="data path")
    parser.add_argument("--output_dir", type=str, default="/data/MaoXiaowei/models/adre/output/adre_sft/",
                        help="output path")
    parser.add_argument("--logdir", type=str, default="/data/MaoXiaowei/models/adre/logs/adre_sft/",
                        help="log directory")
    # training args
    parser.add_argument("--epochs", type=int, default=1, help="epochs")
    parser.add_argument("--micro_batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    # model args
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=4)
    parser.add_argument("--expert_top_k", type=int, default=1)
    parser.add_argument("--adapter_layers", type=int, default=1)
    parser.add_argument("--num_embeddings", type=int, default=1)
    parser.add_argument("--use_hydra_lora", action='store_true', default=True)
    parser.add_argument("--use_multi_lora", action='store_true', default=True)

    args = parser.parse_args()

    # mkdir
    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # config
    config = AutoConfig.from_pretrained(args.model)
    config.num_experts = args.num_experts
    config.lora_rank = args.lora_rank
    config.use_multi_lora = args.use_multi_lora
    config.lora_alpha = args.lora_alpha
    config.use_hydra_lora = args.use_hydra_lora
    config.expert_top_k = args.expert_top_k
    config.adapter_layers = args.adapter_layers
    config.num_embeddings = args.num_embeddings
    # model
    model = AdreQwen2ForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        config=config,
    )

    # frozen model parameters
    trainable_para_name_list = ["adapter", "router_proj", "adre_embedding"]
    for name, param in model.named_parameters():
        if not any(trainable_name in name for trainable_name in trainable_para_name_list):
            param.requires_grad = False
        else:
            param.requires_grad = True

    dataset = load_and_process_data(args.dataset, tokenizer, config.num_experts)

    # 使用自定义数据整理器
    data_collator = AdreDataCollator(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )

    # 5. 训练参数设置
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # deepspeed="./train_config/ds_config.json",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        logging_dir=args.logdir,
        save_steps=args.save_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        fp16=False,
        bf16=True,
        gradient_checkpointing=False,  # AdRE don't support gradient checkpointing
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        lr_scheduler_kwargs={"num_cycles": 0.5},
        optim="adamw_torch",
    )

    indices = list(range(len(dataset)))

    random.shuffle(indices)
    eval_size = min(100, len(dataset))
    train_size = len(dataset) - eval_size

    train_dataset = dataset.select(indices[:train_size])
    eval_dataset = dataset.select(indices[train_size:])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    model.save_pretrained(args.output + 'final_checkpoint/')
    model.config.save_pretrained(args.output + 'final_checkpoint/')
    tokenizer.save_pretrained(args.output + 'final_checkpoint/')
