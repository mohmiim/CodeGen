'''
Created on May 18, 2021

@author: miim
'''
from datasets import load_dataset
import torch
from transformers import  GPT2Tokenizer,GPT2Config, GPT2LMHeadModel, Trainer, DataCollatorForLanguageModeling, TrainingArguments
import os

torch.manual_seed(42)
MAX_LENGTH = 1024

batch_size = 10

CONTINUE = True
device = torch.device("cuda")

def saveModel(model,tokenizer):
    output_dir = './model_save/'
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Saving model to %s" % output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def generate(model,tokenizer):
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

def train(tokenizer,configuration=None,model=None):
    dataset = load_dataset('text',data_files='./data_set.txt')

    # Split into training and validation sets
    train_size = int(0.95 * len(dataset['train']))
    val_size = len(dataset['train']) - train_size
    print('{:>5,} training samples'.format(train_size))
    print('{:>5,} validation samples'.format(val_size))
    
    if (not configuration) :
        configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
    
    def tokenize_function(examples):
        txt = examples["text"];
        return tokenizer("<s> " + txt + " </s>", padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=False)
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(train_size))
     
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False,)
    
    print('Loading model')
    
    if (not model):
        # instantiate the model
        model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
        print('resize_token_embeddings')
        # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
        # otherwise the tokenizer and model tensors won't match up
        model.resize_token_embeddings(len(tokenizer)) 
    
 
    args = TrainingArguments(output_dir="output",
                             per_device_train_batch_size=2,
                             num_train_epochs=5,
                             save_steps=3000,
                             learning_rate = 5e-4,
                             warmup_steps  = 1e2, 
                             adam_epsilon = 1e-8,
                             load_best_model_at_end=True)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=data_collator,
    )
    
    
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    print('The GPT-2 model has {:} different named parameters.\n'.format(len(params)))
    print('==== Embedding Layer ====\n')
    for p in params[0:2]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== First Transformer ====\n')
    for p in params[2:14]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
    print('\n==== Output Layer ====\n')
    for p in params[-2:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
 
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    saveModel(model, tokenizer)
    generate(model,tokenizer)
    
    
if CONTINUE:
    configuration = GPT2Config.from_pretrained('./output_saved')
    model = GPT2LMHeadModel.from_pretrained("./output_saved", config=configuration).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('./output_saved')
    train(tokenizer,configuration,model)  



    
    
        