'''
Created on May 15, 2021

@author: miim
'''

from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2Tokenizer
from HugginfaceDataset import train

FIRST_RUN = True
RUN_TRAINER = False

inputs = ["data_set.txt"]
GPT2_SPECIAL_TOKENs = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token":"<pad>",
    "unk_token":"<unk>",
    "mask_token":"<mask>"
}
 
if FIRST_RUN :
    tokenizer = ByteLevelBPETokenizer()
    special = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>"
        ]
    tokenizer.train(files = inputs, vocab_size=52_000, min_frequency=2,special_tokens=special)
    tokenizer.save_model("tok_output")

test_input = "puts('hello, world')"

#GPT2 tokenizer takes the vocab we generated with the tok 
tokenizer = GPT2Tokenizer.from_pretrained('tok_output')
tokenizer.add_special_tokens(GPT2_SPECIAL_TOKENs)
encoded_input = tokenizer.encode(test_input)
print(encoded_input)
print(tokenizer.decode(encoded_input))
if  RUN_TRAINER:
    train(tokenizer)

