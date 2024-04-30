from dataclasses import dataclass, field
import os
from typing import Optional

@dataclass
class ScriptArguments:

    hf_token: str = field(
        metadata={"help": "Huggingface Token"}
    )

    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"}
    )

    seed: Optional[int] = field(
        default=4761, metadata = {'help':'seed for reproducibility'}
    )

    data_path: Optional[str] = field(
        default="./data/forums_short.json", metadata={"help": "the dataset name"}
    )

    output_dir: Optional[str] = field(
        default="output", metadata={"help": "the output directory"}
    )
    
    per_device_train_batch_size: Optional[int] = field(
        default = 2, metadata = {"help":"the batch size for training"}
    )

    gradient_accumulation_steps: Optional[int] = field(
        default = 1, metadata = {"help":"the gradient accumulation steps for training"}
    )

    optim: Optional[str] = field(
        default = "paged_adamw_32bit", metadata = {"help":"model optimizer"}
    )

    save_steps: Optional[int] = field(
        default = 25, metadata = {"help":"after how many steps do you want to save the model"}
    )

    logging_steps: Optional[int] = field(
        default = 1, metadata = {"help":"the steps after which you start logging model performance"}
    )

    learning_rate: Optional[float] = field(
        default = 2e-4, metadata = {"help":"Model learning rate"}
    )

    max_grad_norm: Optional[float] = field (
        default = 0.3, metadata = {"help":"maximum gradient clipping for training"}
    )

    num_train_epochs: Optional[int] = field (
        default = 1, metadata = {"help":"Number of training epochs"}
    ) 

    warmup_ratio: Optional[float] = field (
        default = 0.03, metadata = {"help":"ratio of steps to use for warming up gradient"}
    )

    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata = {"help":"learning rate scheduler"}
    ) 

    lora_dir: Optional[str] = field(
            default = "./model/llm_hate_speech_lora", metadata = {"help":"the directory in which to save the LoRA model"}
    )

    max_steps: Optional[int] = field(
            default=-1, metadata={"help": "the number of training steps"}
    )

