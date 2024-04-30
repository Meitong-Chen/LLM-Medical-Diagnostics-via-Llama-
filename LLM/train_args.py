from dataclasses import dataclass, field
import os
from typing import Optional

@dataclass
class ScriptArguments:

    hf_token: str = field(metadata={"help": ""})


    model_name: Optional[str] = field(
        default="meta-llama/Llama-2-7b-hf", metadata={"help": ""}
    )

    seed: Optional[int] = field(
        default=4761, metadata = {'help':''}
    )

    data_path: Optional[str] = field(
        default="./data/forums_short.json", metadata={"help": ""}
    )

    output_dir: Optional[str] = field(
        default="output", metadata={"help": ""}
    )
    
    per_device_train_batch_size: Optional[int] = field(
        default = 2, metadata = {"help":""}
    )

    gradient_accumulation_steps: Optional[int] = field(
        default = 1, metadata = {"help":""}
    )

    optim: Optional[str] = field(
        default = "paged_adamw_32bit", metadata = {"help":""}
    )

    save_steps: Optional[int] = field(
        default = 25, metadata = {"help":""}
    )

    logging_steps: Optional[int] = field(
        default = 1, metadata = {"help":""}
    )

    learning_rate: Optional[float] = field(
        default = 2e-4, metadata = {"help":""}
    )

    max_grad_norm: Optional[float] = field (
        default = 0.3, metadata = {"help":""}
    )

    num_train_epochs: Optional[int] = field (
        default = 1, metadata = {"help":""}
    ) 

    warmup_ratio: Optional[float] = field (
        default = 0.03, metadata = {"help":""}
    )

    lr_scheduler_type: Optional[str] = field(
        default="cosine", metadata = {"help":""}
    ) 

    lora_dir: Optional[str] = field(default = "./model/llm_hate_speech_lora", metadata = {"help":""})

    max_steps: Optional[int] = field(default=-1, metadata={"help": ""})

    text_field: Optional[str] = field(default='chat_sample', metadata={"help": ""})


