import unittest

import torch
from transformers import AutoTokenizer

from adre.models.adre_qwen2 import AdreQwen2ForCausalLM


class TestAdreQwen2(unittest.TestCase):

    def test_load_adre_qwen2(self):
        # 从已有的 AdreQwen2 checkpoint 加载
        from transformers import Qwen2Config

        config = Qwen2Config.from_pretrained("/data/MaoXiaowei/models/sier/model/")
        config.num_experts = 2
        config.lora_rank = 8
        config.use_multi_lora = True
        config.lora_alpha = 8
        config.use_hydra_lora = True
        config.expert_top_k = 1
        config.adapter_layers = 3
        config.num_embeddings = 4

        model = AdreQwen2ForCausalLM.from_pretrained(
            "/data/MaoXiaowei/models/sier/model/",
            config=config
        ).cuda()
        tokenizer = AutoTokenizer.from_pretrained("/data/MaoXiaowei/models/sier/model/")

        # generation
        message = [{"role": "user", "content": "hello!"}]
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs_ids = inputs.input_ids.cuda()

        outputs = model.generate(inputs_ids, max_length=100)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print(response)

        # specify gate values
        gate_values = torch.tensor([[1.0, 0.0]]).cuda()
        outputs = model.generate(inputs_ids, max_length=100, gate_values=gate_values)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print(response)

        # set use_adapter to False
        outputs = model.generate(inputs_ids, max_length=100, use_adapter=False)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        print(response)

        inputs = tokenizer([prompt + '</think>', prompt + '</think>\nhello,'],
                           return_tensors="pt",
                           padding=True,
                           truncation=True)
        inputs_ids = inputs.input_ids.cuda()
        output_mask = torch.zeros(size=(2, inputs_ids.shape[1])).cuda()
        output_mask[:, -1] = 1
        outputs = model.generate(inputs_ids, max_length=100,
                                 output_mask=output_mask,
                                 attention_mask=inputs.attention_mask.cuda())
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(response)
