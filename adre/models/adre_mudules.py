from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRABlock(nn.Module):
    def __init__(self,
                 in_features, out_features, config, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.rank = config.lora_rank
        self.scaling = config.lora_alpha / (config.lora_rank ** 0.5)
        self.lora_a = nn.Linear(self.in_features, self.rank, bias=False, device=device, dtype=dtype)
        self.lora_b = nn.Linear(self.rank, out_features, bias=False, device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.lora_b.weight)

    def forward(self,
                hidden_states: torch.Tensor,
                gate_values: Optional[torch.Tensor] = None):
        return self.lora_b(self.lora_a(hidden_states)) * self.scaling


class MultiLoRABlock(nn.Module):
    def __init__(self,
                 in_features, out_features, config, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = config.num_experts

        self.rank = config.lora_rank
        self.scaling = config.lora_alpha / (config.lora_rank ** 0.5)

        self.use_hydra_lora = config.use_hydra_lora
        if config.use_hydra_lora:
            self.lora_a = nn.Linear(self.in_features, self.rank, bias=False, device=device, dtype=dtype)
        else:
            self.lora_a = nn.Linear(self.in_features, self.num_experts * self.rank, bias=False, device=device,
                                    dtype=dtype)

        self.lora_b = nn.Linear(self.rank * self.num_experts, self.out_features, bias=False, device=device, dtype=dtype)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.lora_b.weight)

    def forward(self,
                hidden_states: torch.Tensor,
                gate_values: torch.Tensor):
        """

        :param hidden_states: size = (bsz, -1, in_features)
        :param gate_values: size = (bsz, num_expert)
        :return:
        """
        x = self.lora_a(hidden_states)  # (bsz, -1, num_expert*rank)
        shape = (*x.shape[:-1], self.num_experts, self.rank)  # (bsz, -1, num_expert, rank)
        if self.use_hydra_lora:
            x = x.unsqueeze(-2).expand(shape)
        else:
            x = x.view(shape)
        gate_values = gate_values.unsqueeze(-2).expand(shape[:-1]).unsqueeze(-1)
        x = (x * gate_values).view(*shape[:-2], -1)
        x = self.lora_b(x) * self.scaling
        return x


class AdapterLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, config, use_multi_lora=True, bias: bool = True,
                 device=None, dtype=None):
        super().__init__(in_features, out_features, bias,
                         device, dtype)
        self.use_multi_lora = use_multi_lora
        if self.use_multi_lora:
            self.adapter = MultiLoRABlock(in_features, out_features, config, device, dtype)
        else:
            self.adapter = LoRABlock(in_features, out_features, config, device, dtype)

    def forward(self,
                hidden_states: torch.Tensor,
                gate_values: Optional[torch.Tensor] = None,
                use_adapter: bool = True):
        if not use_adapter:
            # don't use lora
            return F.linear(hidden_states, self.weight, self.bias)
        else:
            # use lora
            return F.linear(hidden_states, self.weight, self.bias) + self.adapter(hidden_states, gate_values)
