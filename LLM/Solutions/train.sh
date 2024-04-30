#!/bin/bash
python3 -m venv llm_env
source llm_env/bin/activate
pip install -r requirements.txt
python3 train.py --model_name mistralai/Mistral-7B-Instruct-v0.2 \
                --data_path jbrophy123/quora_dataset \
                --save_steps 100000 \
                --per_device_train_batch_size 1 \
                --lora_dir jbrophy123/quora_llm \
                --hf_token <token> \
                --max_steps 2000
