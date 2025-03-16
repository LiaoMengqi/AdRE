import unittest

import torch
from transformers import Qwen2Config

from adre.models.adre_qwen2 import AdreQwen2MLP


class TestAdreQwen2MLP(unittest.TestCase):
    def setUp(self):
        self.config = Qwen2Config(
            hidden_size=32,
            intermediate_size=128,
        )
        self.config.num_experts = 2
        self.config.lora_rank = 8
        self.config.use_multi_lora = False
        self.config.lora_alpha = 8
        self.config.use_hydra_lora = False
        self.config.top_k = 1

        self.mlp = AdreQwen2MLP(self.config)

    def test_output_shape(self):
        batch_size = 2
        seq_length = 10
        hidden_size = self.config.hidden_size
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        gate_values = torch.randn(batch_size, self.config.num_experts)
        
        output = self.mlp(hidden_states, gate_values)
        
        self.assertEqual(output.shape, (batch_size, seq_length, hidden_size))

    def test_train(self):
        batch_size = 2
        seq_length = 10
        hidden_size = self.config.hidden_size
        
        hidden_states = torch.randn(batch_size, seq_length, hidden_size)
        gate_values = torch.randn(batch_size, self.config.num_experts)
        
        self.mlp.train()
        output = self.mlp(hidden_states, gate_values)
        output.sum().backward()
        
        for name, param in self.mlp.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} should have gradients")
            self.assertFalse(torch.isnan(param.grad).any(), f"{name} has NaN gradients")

    def test_top_k_gate_values(self):
        batch_size = 2
        gate_values = torch.tensor([[0.1, 0.8, 0.3], 
                                  [0.2, 0.4, 0.9]], dtype=torch.float32)
        
        binary_gates = self.mlp.get_top_k_gate_values(gate_values)
        
        # 验证每行只有一个1(因为top_k=1)
        self.assertEqual(binary_gates.sum().item(), 2.0)
        # 验证选择的是最大值
        self.assertEqual(binary_gates[0, 1].item(), 1.0)  # 第一行最大值在索引1
        self.assertEqual(binary_gates[1, 2].item(), 1.0)  # 第二行最大值在索引2
