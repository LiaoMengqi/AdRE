import glob
import torch
from transformers import AutoTokenizer, AutoConfig
from adre.models.adre_qwen2 import AdreQwen2ForCausalLM
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 基本参数
    parser.add_argument("--model", type=str, default="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-mtai/users/liaomengqi/model/adre_test/checkpoint-800/", help="模型路径")
    # parser.add_argument("--dataset", type=str, default="/path/to/dataset", help="数据集路径") 
    # parser.add_argument("--output", type=str, default="/path/to/output", help="输出路径")
    # 模型参数
    args = parser.parse_args()
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    
    # 加载模型
    model = AdreQwen2ForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16
    ).cuda()
    model.eval()
    
    # 测试句子
    test_text = "2*19-9^2=？"
    
    # 编码输入
    inputs = tokenizer.apply_chat_template([{"role":"user","content":test_text}], tokenize=False, add_generation_prompt=True)
    inputs = inputs if inputs.endswith("<think>\n") else inputs + "<think>\n" 

    print(inputs)
    model_inputs = tokenizer(inputs, return_tensors="pt")

    
    # 设置不同的gate_values进行测试
    gate_values_list = [
        [1, 0, 0, 0, 0, 0, 0, 0],  # 使用第1个专家
        [0, 1, 0, 0, 0, 0, 0, 0],  # 使用第2个专家
        [0, 0, 1, 0, 0, 0, 0, 0],  # 使用第3个专家
        [0, 0, 0, 1, 0, 0, 0, 0],  # 使用第4个专家
        [0, 0, 0, 0, 1, 0, 0, 0],  # 使用第5个专家
        [0, 0, 0, 0, 0, 1, 0, 0],  # 使用第6个专家
        [0, 0, 0, 0, 0, 0, 1, 0],  # 使用第7个专家
        [0, 0, 0, 0, 0, 0, 0, 1],  # 使用第8个专家
    ]
    
    print("测试句子:", test_text)
    print("\n不同专家的生成结果:")
    
    # 准备batch输入
    batch_size = len(gate_values_list)
    input_ids = model_inputs.input_ids.repeat(batch_size, 1).cuda()
    attention_mask = model_inputs.attention_mask.repeat(batch_size, 1).cuda()
    gate_values = torch.tensor(gate_values_list, dtype=torch.bfloat16).cuda()

    # 测试 ref_logits
    with_ref_output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        gate_values=gate_values,
        caculate_ref=True
    )

    output=model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        gate_values=gate_values,
        caculate_ref=False
    )
    
    # batch生成
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        gate_values=gate_values,
        max_new_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=False,
        # use_adapter=False
    )
    
    # 解码并打印每个专家的输出
    for i, output in enumerate(outputs):
        generated_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"\n专家 {i+1} 的生成结果:")
        print(generated_text)
