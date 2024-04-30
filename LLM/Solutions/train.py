import os
#import packages
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    HfArgumentParser
)
from datasets import load_dataset
import torch

import bitsandbytes as bnb
from huggingface_hub import login, HfFolder

from trl import SFTTrainer

from utils import print_trainable_parameters, find_all_linear_names

from train_args import ScriptArguments

from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, prepare_model_for_kbit_training

#parse training arguments
parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]
#define training function
def training_function(args):
    #Huggingface login
    login(token=args.hf_token)

    # set seed
    set_seed(args.seed)

    #specify data path and load in data
    data_path=args.data_path

    dataset = load_dataset(data_path)

    #create bits and bytes configuration for 4 bit model quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, #load in 4 bit quantization
        bnb_4bit_use_double_quant=True, # use double quantization to quantize the quantization constants
        bnb_4bit_quant_type="nf4", # unique int type for quantizing model
        bnb_4bit_compute_dtype=torch.bfloat16, #This sets the computational type which might be different than the input time. For example, inputs might be fp32, but computation can be set to bf16 for speedups.
    )
    #create new object to store pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, #specify model
        use_cache=False, #don't store it locally to save memory
        device_map="auto", #automatically find available GPU's
        quantization_config=bnb_config, #set quantization config
        trust_remote_code=True #this allows us to pull in models from HuggingFace
    )

    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    #set padding token to the end of sequence token
    tokenizer.pad_token=tokenizer.eos_token
    #set padding side
    tokenizer.padding_side='right'

    #This preps the model for 4 bit training
    model=prepare_model_for_kbit_training(model)
    #use utils function to find all model layers for which we will be applying LoRA
    modules=find_all_linear_names(model)
    #set LoRA Config
    config = LoraConfig(
        r=64, #The rank of our new matrices
        lora_alpha=16, #the alpha parameter for LoRA scaling
        lora_dropout=0.1, #dropout hyperparameter to prevent overfitting
        bias='none', #no preset bias term
        task_type='CAUSAL_LM', #Our task is called 'Causal Language Modelling' because we are predicting tokens (it's not technically causal but that's what they call it)
        target_modules=modules #The model layers on which we will apply LoRA
    )
    #convert the model to a peft model, allowing us to use LoRA
    model=get_peft_model(model, config)
    # Define training args
    output_dir = args.output_dir
    training_arguments = TrainingArguments(
        output_dir=output_dir, #directory to save intermediate steps
        per_device_train_batch_size=args.per_device_train_batch_size, #number of observations per batch per device
        gradient_accumulation_steps=args.gradient_accumulation_steps, # Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
        optim=args.optim, # Gradient Descent Optimizer
        save_steps=args.save_steps, # after how many steps to save an intermediate copy of the model
        logging_steps=args.logging_steps, # This is for logging model error, but we don't use this
        learning_rate=args.learning_rate, # training learning rate
        bf16=False, # Whether to use bfloat16 int type
        max_grad_norm=args.max_grad_norm, #Maximum gradient norm (for gradient clipping).
        num_train_epochs=args.num_train_epochs, #Number of training epochs
        warmup_ratio=args.warmup_ratio, #Ratio of total training steps used for a linear warmup from 0 to learning_rate.
        group_by_length=True, #Batch observations of similar sizes together to speed up training
        lr_scheduler_type=args.lr_scheduler_type, #Learning scheduler
        tf32=False, #whether to use tensorfloat in computation, we don't need it
        report_to="none", #don't report training error (for simplicity)
        push_to_hub=False, # we manually push to huggingface below so no need to do it here
        max_steps = args.max_steps #Max number of batches the model will train on
    )
    #Create a new class of supervised fine tuning trainer
    trainer = SFTTrainer(
        model=model, #specify model
        train_dataset=dataset['train'], #training data
        dataset_text_field="text", #column with the training observations
        max_seq_length=2048, #maximum sequence length to truncate new values
        tokenizer=tokenizer, #what tokenizer to use
        args=training_arguments #specify training arguments
    )

    #This converts all normalization layers to float32 int type for more stable training
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    print('starting training')
    #start training
    trainer.train()

    print('LoRA training complete')
    lora_dir = args.lora_dir
    #push model to huggingface
    trainer.model.push_to_hub(lora_dir, safe_serialization=False)
    
    print("saved lora adapters")

    
#run function
if __name__=='__main__':
    training_function(args)

