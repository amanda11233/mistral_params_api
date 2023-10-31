from datasets import load_dataset
from random import randrange
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer


dataset = load_dataset("json", data_files={'train':'p2.json'})

dataset = dataset['train']
def format_instruction(sample):
	return f"""### Instruction:
From the Instruction below, extract the parameters and generate the output in JSON Response.

### Input:
{sample['input']}

### JSON Resposne:
{sample['output']}
"""


use_flash_attention = False

# Hugging Face model id
model_id = "mistralai/Mistral-7B-Instruct-v0.1"  # non-gated
# model_id = "meta-llama/Llama-2-7b-hf" # gated


# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    use_cache=False,
    use_flash_attention_2=use_flash_attention,
    device_map="auto",
)
model.config.pretraining_tp = 1


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA config based on QLoRA paper
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
"q_proj",
"k_proj",
"v_proj",
"o_proj",
"gate_proj",
"up_proj",
"down_proj",
"lm_head",
],
        )


# prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)
 


args = TrainingArguments(
    output_dir="Mistral-7-int4-dolly",
    num_train_epochs=20,
    per_device_train_batch_size=2 if use_flash_attention else 4,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=0.0001,
    bf16=True,
    tf32=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    disable_tqdm=False # disable tqdm since with packing values are in correct
)



max_seq_length = 2048 # max sequence length for model and packing of the dataset

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=format_instruction,
    args=args,
)

# train
trainer.train() # there will not be a progress bar since tqdm is disabled

# save model
trainer.save_model()
