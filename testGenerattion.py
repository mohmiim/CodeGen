'''
Created on May 21, 2021

@author: miim
'''
import torch
from transformers import  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

torch.manual_seed(55)
device = torch.device("cuda")

configuration = GPT2Config.from_pretrained('./output')
model = GPT2LMHeadModel.from_pretrained("./output", config=configuration).to(device)
tokenizer = GPT2Tokenizer.from_pretrained('./output')

prompt = "<s> "
generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
generated = generated.to(device)
print(generated)
sample_outputs = model.generate(
                        generated, 
                        #bos_token_id=random.randint(1,30000),
                        do_sample=True,   
                        top_k=50, 
                        max_length = 300,
                        top_p=0.95, 
                        num_return_sequences=3
                        )

for i, sample_output in enumerate(sample_outputs):
    output = tokenizer.decode(sample_output, skip_special_tokens=True)
    output  = output.replace('<N>','\n')
    print("{}: {}\n\n".format(i, output))
    print("===============================")
