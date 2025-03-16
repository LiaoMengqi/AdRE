import unittest

import torch
from transformers import Qwen2Config

from adre.models.adre_qwen2 import AdreQwen2Attention


class TestAdreQwen2Attention(unittest.TestCase):
    def setUp(self):
        self.config = Qwen2Config(
            hidden_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        self.config.num_embeddings = 3
        self.config.num_experts = 2
        self.config.lora_rank=8
        self.config.use_multi_lora = True
        self.config.lora_alpha = 8
        self.config.use_hydra_lora = True
        self.config.top_k = 1
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads

        self.attention = AdreQwen2Attention(self.config, 0)
    def get_atten_mask(self, seq_length: torch.Tensor, max_seq_len):
        # seq_length is a tensor of shape (batch_size, 1)
        # return a tensor of shape (batch_size, 1, seq_length, seq_length)
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
        # 输出长度小于 seq_length，将输出部分设置为 1， 输入和 padding 为 0，返回shape (bsz,max_seq_length)，left padding 
        batch_size = seq_length.shape[0]
        mask = torch.zeros(batch_size, max_seq_length)
        for i in range(batch_size):
            # 计算输出长度为序列长度的一半
            output_len = seq_length[i].item() // 2
            # 从右到左填充1,实现left padding
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
        output = self.attention(**sample_inputs)

        self.assertEqual(output[0].shape, (batch_size, seq_length, self.config.hidden_size))
        self.assertEqual(output[1].shape, (batch_size, self.config.num_attention_heads, seq_length, seq_length))
        self.assertEqual(output[2].shape, (batch_size, self.config.num_experts))

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
    
        self.attention.train()
        output = self.attention(**sample_inputs)
        output[0].sum().backward(retain_graph=True)
        output[2].sum().backward()

        for name, param in self.attention.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} should have gradients")
            self.assertFalse(torch.isnan(param.grad).any(), f"{name} has NaN gradients")

    # def test_gradient_flow(self):
    #     output = self.attention(sample_inputs)
    #     output[0].sum().backward()
    #
    #     for name, param in self.attention.named_parameters():
    #         self.assertIsNotNone(param.grad, f"{name} should have gradients")
    #         self.assertFalse(torch.isnan(param.grad).any(), f"{name} has NaN gradients")

    # def test_attention_cache(self):
    #     batch_size = 2
    #     seq_length = 8
    #     past_length = 4
    #     hidden_size = self.config.hidden_size
    #
    #     inputs = {
    #         'hidden_states': torch.randn(batch_size, seq_length, hidden_size),
    #         'attention_mask': torch.ones(batch_size, 1, seq_length + past_length, seq_length + past_length),
    #         'position_ids': None,
    #         'past_key_value': (
    #             torch.randn(batch_size, past_length, hidden_size),
    #             torch.randn(batch_size, past_length, hidden_size)
    #         ),
    #         'use_cache': True
    #     }
    #
    #     output = self.attention(**inputs)
    #
    #     self.assertEqual(len(output), 3, "Should return output, attention weights and cache")
    #     self.assertEqual(len(output[2]), 2, "Cache should contain key and value")
    #     self.assertEqual(output[2][0].shape, (batch_size, seq_length + past_length, hidden_size))
    #     self.assertEqual(output[2][1].shape, (batch_size, seq_length + past_length, hidden_size))

    # def test_different_input_sizes(self):
    #     test_cases = [
    #         (1, 8),
    #         (2, 16),
    #         (4, 32)
    #     ]
    #
    #     for batch_size, seq_length in test_cases:
    #         with self.subTest(batch_size=batch_size, seq_length=seq_length):
    #             inputs = {
    #                 'hidden_states': torch.randn(batch_size, seq_length, self.config.hidden_size),
    #                 'attention_mask': torch.ones(batch_size, 1, seq_length, seq_length),
    #                 'position_ids': None
    #             }
    #
    #             output = self.attention(**inputs)
    #             self.assertEqual(output[0].shape, (batch_size, seq_length, self.config.hidden_size))
    #
    # def test_input_validation(self):
    #     with self.assertRaisesRegex(ValueError, "hidden_states shape"):
    #         self.attention(
    #             hidden_states=torch.randn(2, 8),  # Missing hidden_size dimension
    #             attention_mask=torch.ones(2, 1, 8, 8),
    #             position_ids=None
    #         )
    #
    #     with self.assertRaisesRegex(ValueError, "attention_mask shape"):
    #         self.attention(
    #             hidden_states=torch.randn(2, 8, 32),
    #             attention_mask=torch.ones(2, 8),  # Wrong shape
    #             position_ids=None
    #         )
