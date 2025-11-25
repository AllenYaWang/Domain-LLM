import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForSeq2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# 1. 配置路径和参数
model_id = "models/Qwen/Qwen2.5-7B"  # 请替换为你本地模型的路径或HuggingFace ID
output_dir = "models/Qwen/qwen2.5-ner-sft"

# 2. 加载数据
dataset = load_dataset("json", data_files={"train": "./datasets/train.jsonl", "validation": "./datasets/val.jsonl"})

# 3. 加载Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 4. 加载模型 (使用4bit量化加载以节省显存，如果显存足够可去掉 load_in_4bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
# 打印设备映射
print("="*20 + "device of mode map" + "="*20)
print("Device Map:", model.hf_device_map)

# 5. 配置 LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 6. 训练参数
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    fp16=True,
    optim="paged_adamw_32bit",
    report_to="none" # 不上报wandb
)

# 7. 开始训练
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()
trainer.save_model(output_dir)
print("Training completed and model saved.")
