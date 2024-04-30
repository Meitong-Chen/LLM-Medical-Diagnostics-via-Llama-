import os

import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
import torch

import bitsandbytes as bnb
from huggingface_hub import login, HfFolder

import torch.nn as nn

# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    trainable_params /= 2
    print(
        f"all params: {all_param:,d} || trainable params: {trainable_params:,d} || trainable%: {100 * trainable_params / all_param}"
    )

def format_prompt(eval_str):
    prompt=f"""###User: Categorize the following statement as offensive, neutral, or hateful: \n\n'{eval_str}'\n
follow this format with your response:

(1) Holistic Explanation: Provide a brief explanation (up to 150 words) on why the statement falls into the chosen category. If neutral, explain why it's neutral. If it's offensive or hateful, summarize the harmful impact,  detailing its impact on each group mentioned.

Classification: (neutral, offensive, hate)

If the statement is categorized as offensive or hateful, also include the following:

(2) Affected Groups: List the group(s) most harmed by the statement.
(3) Harmful Words: Identify specific harmful word(s) or phrase(s) directed at each group.
(4) Reason for Harm: Briefly explain why the words are hurtful to each group.

### Assistant:
"""
    return prompt

class Multiclass(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(384, 8)
        self.act = nn.ReLU()
        self.output = nn.Linear(8, 3)
        
    def forward(self, x):
        x = self.act(self.hidden(x))
        x = self.output(x)
        return x


# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)
