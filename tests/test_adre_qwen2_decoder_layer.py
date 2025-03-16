import unittest

import torch
from transformers import Qwen2Config

from adre.models.adre_qwen2 import AdreQwen2DecoderLayer


class TestAdreQwen2DecoderLayer(unittest.TestCase):
    def setUp(self):
        self.config = Qwen2Config(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=64
        )
        self.config.num_embeddings = 3
        self.config.num_experts = 8
        self.config.lora_rank = 8
        self.config.use_multi_lora = True
        self.config.lora_alpha = 8
        self.config.use_hydra_lora = True
        self.config.top_k = 1
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads

        self.decoder_layer = AdreQwen2DecoderLayer(self.config, 0)

    def get_atten_mask(self, seq_length: torch.Tensor, max_seq_len):
        batch_size = seq_length.shape[0]
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        mask = (mask == 1).float()
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.repeat(batch_size, 1, 1, 1)

        for i in range(batch_size):
            mask[i, :, :, :max_seq_len - seq_length[i].item()] = 0  # pad
            mask[i, :, :max_seq_len - seq_length[i].item(), :] = 0
        mask = (1 - mask) * torch.finfo(mask.dtype).min
        return mask

    def get_out_put_mask(self, seq_length: torch.Tensor, max_seq_length):
        batch_size = seq_length.shape[0]
        mask = torch.zeros(batch_size, max_seq_length)
        for i in range(batch_size):
            output_len = seq_length[i].item() // 2
            mask[i, max_seq_length-output_len:max_seq_length] = 1
        return mask

    def test_output_shape(self):
        batch_size = 2
        max_seq_length = 10
        hidden_size = self.config.hidden_size
        seq_len = torch.randint(4, max_seq_length, (batch_size, 1))
        seq_length = seq_len.max().item()

        sample_inputs = {
            'hidden_states': torch.randn(batch_size, seq_length, hidden_size),
            'attention_mask': self.get_atten_mask(seq_len, seq_length),
            'position_embeddings': (torch.randn(batch_size, seq_length, self.head_dim),
                                  torch.randn(batch_size, seq_length, self.head_dim))
        }
        output = self.decoder_layer(**sample_inputs,use_adapter=False)

        self.assertEqual(output[0].shape, (batch_size, seq_length, self.config.hidden_size))

    def test_train(self):
        batch_size = 2
        max_seq_length = 10
        hidden_size = self.config.hidden_size
        seq_len = torch.randint(4, max_seq_length, (batch_size, 1))
        seq_length = seq_len.max().item()

        sample_inputs = {
            'hidden_states': torch.randn(batch_size, seq_length, hidden_size),
            'attention_mask': self.get_atten_mask(seq_len, seq_length),
            'position_embeddings': (torch.randn(batch_size, seq_length, self.head_dim),
                                  torch.randn(batch_size, seq_length, self.head_dim)),
            'output_mask': self.get_out_put_mask(seq_len, seq_length)
        }

        self.decoder_layer.train()
        output = self.decoder_layer(**sample_inputs)
        output[0].sum().backward(retain_graph=True)

        for name, param in self.decoder_layer.named_parameters():
            if 'router_proj' in name:
                continue
            self.assertIsNotNone(param.grad, f"{name} should have gradients")
            self.assertFalse(torch.isnan(param.grad).any(), f"{name} has NaN gradients")
