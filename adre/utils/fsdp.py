import functools

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer


def get_fsdp_wrapper(model, precision="bf16", sharding_strategy="FULL_SHARD"):
    """
    Wrap the model as an FSDP model
    
    Args:
        model: The model to be wrapped
        precision: Precision setting, supports "bf16", "fp16", "fp32"
        sharding_strategy: Sharding strategy, supports "FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"
    
    Returns:
        The wrapped FSDP model
    """
    # 设置混合精度
    if precision == "bf16":
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    elif precision == "fp16":
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    else:  # fp32
        mixed_precision_policy = None
    
    # 设置分片策略
    if sharding_strategy == "FULL_SHARD":
        strategy = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "SHARD_GRAD_OP":
        strategy = ShardingStrategy.SHARD_GRAD_OP
    else:  # NO_SHARD
        strategy = ShardingStrategy.NO_SHARD
    
    auto_wrap_policy = None
    if hasattr(model, "config"):
        def trainable_params_auto_wrap_policy(module, recurse=True, **kwargs):
            # 检查模块是否包含可训练参数
            has_trainable_params = any(p.requires_grad for p in module.parameters())
            
            if model.config.model_type == "llama":
                from ..models.adre_llama import AdreLlamaDecoderLayer
                # 如果是目标层类型且包含可训练参数，则进行包装
                if isinstance(module, (LlamaDecoderLayer, AdreLlamaDecoderLayer)) and has_trainable_params:
                    return True
            elif model.config.model_type == "qwen2":
                from ..models.adre_qwen2 import AdreQwen2DecoderLayer
                # 如果是目标层类型且包含可训练参数，则进行包装
                if isinstance(module, (Qwen2DecoderLayer, AdreQwen2DecoderLayer)) and has_trainable_params:
                    return True
            
            # 对于其他模块，只有当它们包含可训练参数且参数量足够大时才包装
            if has_trainable_params and sum(p.numel() for p in module.parameters() if p.requires_grad) > 1e6:
                return True
            
            return False
        
        auto_wrap_policy = trainable_params_auto_wrap_policy
    
    if auto_wrap_policy is None:
        auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=1e8)
    
    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=strategy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device(),
        cpu_offload=CPUOffload(offload_params=False),
    )
    
    return fsdp_model
