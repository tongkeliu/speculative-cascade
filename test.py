from modelscope import AutoTokenizer, AutoModelForCausalLM
from transformers import DynamicCache
import torch
from utils import *


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    dtype="auto",
    device_map="cuda:0",
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="cuda:0",
    trust_remote_code=True
)

prompt = "who are you?"
past_key_values = DynamicCache()
max_length = 256
generated_tokens = []

for i in range(max_length):
    if generated_tokens == []:
        input_ids = tokenizer(prompt, return_tensors="pt").to(device)
        output = model(**input_ids, use_cache=True, past_key_values=past_key_values)
    else:
        output = model(input_ids=sampled_token, use_cache=True, past_key_values=past_key_values)

    logits = norm_top_k_top_p_filter(0.7, output.logits[:,-1,:], 20, 0.9)
    sampled_token = torch.multinomial(logits, num_samples=1)
    generated_tokens.append(sampled_token.item())

sentence = tokenizer.decode(generated_tokens)
print(sentence)