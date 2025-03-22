
from transformers.models.llama.modeling_llama import LlamaForCausalLM,LlamaDecoderLayer

class AdreLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

class AdreLLaMAForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, **kwargs):
        return super().forward(input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, **kwargs)
