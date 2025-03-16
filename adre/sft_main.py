import glob
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments, Trainer, AutoConfig, )

from .models.adre_qwen2 import AdreQwen2ForCausalLM


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
    output_mask = []
    gate_values = []
    for i in range(batch_size):
        label = [-100] * (input_lengths[i]) + tokenized_outputs["input_ids"][i]
        labels.append(label)
        input_ids.append(tokenized_inputs["input_ids"][i] + tokenized_outputs["input_ids"][i])
        output_mask.append([0] * (input_lengths[i]) + [1] * output_mask[i])
        gate_value = [0] * num_experts
        gate_value[int(examples['rank'][i] * num_experts - 1e-5)] = 1
        gate_values.append(gate_value)
    model_inputs = {'input_ids': input_ids,
                    'labels': labels,
                    'input_len': input_lengths,
                    'target_length': output_lengths,
                    'output_mask': output_mask,
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
        for feature in features:
            orig_position_ids = list(range(feature["input_len"] + feature["target_length"]))

            padding_length = max_length - len(orig_position_ids)
            if self.tokenizer.padding_side == "right":
                padded_ids = orig_position_ids + [0] * padding_length
            else:
                padded_ids = [0] * padding_length + orig_position_ids
            padded_position_ids.append(padded_ids)

        batch["position_ids"] = torch.tensor(padded_position_ids, dtype=torch.long, device=batch["input_ids"].device)
        return batch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # general args
    parser.add_argument("--model", type=str, default="./model/", help="model name or path")
    parser.add_argument("--dataset", type=str, default="./data/extend_data_18k/", help="data path")
    parser.add_argument("--output_dir", type=str, default="./output/", help="output path")
    parser.add_argument("--logdir", type=str, default="./logs/", help="log directory")
    # training args
    parser.add_argument("--epochs", type=int, default=1, help="epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    # model args
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--expert_top_k", type=int, default=1)
    parser.add_argument("--adapter_layers", type=int, default=4)
    parser.add_argument("--num_embeddings", type=int, default=4)
    parser.add_argument("--use_hydra_lora", action='store_true', default=False)
    parser.add_argument("--use_multi_lora", action='store_true', default=False)

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
        torch_dtype=torch.bfloat16
    )

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
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        logging_dir=args.logdir,
        save_steps=200,
        evaluation_strategy="steps",
        eval_steps=200,
        save_total_limit=1,
        load_best_model_at_end=True,  # 加载最佳模型
        metric_for_best_model="eval_loss",  # 以验证集损失为指标
        greater_is_better=False,
        report_to="tensorboard",
        fp16=False,
        bf16=True,  # 如果你的GPU支持bf16，可以设为True
        gradient_checkpointing=True,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",  # 使用余弦学习率衰减
        lr_scheduler_kwargs={"num_cycles": 0.5},  # 半个余弦周期
    )

    train_size = len(dataset) - 100
    eval_size = min(100, len(dataset))
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(args.output + '/final_checkpoint/')
    model.config.save_pretrained(args.output + '/final_checkpoint/')
    tokenizer.save_pretrained(args.output + '/final_checkpoint/')
