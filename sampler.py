import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from utils import *

model_collection = {
    "qwen3-0.6b":"Qwen/Qwen3-0.6B",
    "qwen3-4b":"Qwen/Qwen3-4B",
    "qwen3-8b":"Qwen/Qwen3-8B"
}

class Sampler():
    def __init__(self, args):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_collection[args.approx_model],
            dtype="auto",
            device_map=args.device,
            trust_remote_code=True  
        )
        self.approx_model = AutoModelForCausalLM.from_pretrained(
            model_collection[args.approx_model],
            dtype="auto",
            device_map=args.device,
            trust_remote_code=True  
        )
        self.target_model = AutoModelForCausalLM.from_pretrained(
            model_collection[args.target_model],
            dtype="auto",
            device_map=args.device,
            trust_remote_code=True          
        )
    
    @torch.no_grad()
    def autoregressive(self):
        model = self.target_model
        past_key_values = None
        generated_tokens = []

        for i in range(self.args.max_length):
            if past_key_values == None:
                input_ids = self.tokenizer(self.args.prompt, return_tensors="pt").to(self.args.device)
                output = model(**input_ids, use_cache=True)
            else:
                output = model(input_ids=sampled_token, use_cache=True, past_key_values=past_key_values)

            past_key_values = output.past_key_values
            logits = norm_top_k_top_p_filter(self.args.temperature, output.logits[:,-1,:],\
                                             self.args.top_k, self.args.top_p)
            sampled_token = torch.multinomial(logits, num_samples=1)
            generated_tokens.append(sampled_token.item())
        
        return generated_tokens
    
    def speculative(self):
        pass

    def cascade(self):
        pass

    def spec_cascade(self):
        pass
