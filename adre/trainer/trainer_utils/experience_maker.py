import torch
from dataclasses import dataclass
from typing import Optional, Union


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Samples:
    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    case: Optional[list[int]]


@dataclass
class Experience:
    sequences: torch.Tensor
    old_log_probs: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    ref_log_probs: Optional[torch.Tensor]
    info: Optional[dict]

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.old_log_probs = to(self.old_log_probs, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.ref_log_probs = to(self.ref_log_probs, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self

