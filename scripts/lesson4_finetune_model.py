# Citation of TinyLlama
# @misc{zhang2024tinyllama,
#       title={TinyLlama: An Open-Source Small Language Model}, 
#       author={Peiyuan Zhang and Guangtao Zeng and Tianduo Wang and Wei Lu},
#       year={2024},
#       eprint={2401.02385},
#       archivePrefix={arXiv},
#       primaryClass={cs.CL}
# }

from unsloth import FastLanguageModel

# 下載 & 載入模型（用 Unsloth 最佳化版本）
model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "TinyLlama/TinyLlama_v1.1_Chinese",
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-V0.4",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)



## URI : https://medium.com/@bohachu/%E7%B9%81%E9%AB%94%E4%B8%AD%E6%96%87%E7%9A%84-llama-3-%E7%B9%BC%E7%BA%8C%E9%A0%90%E8%A8%93%E7%B7%B4continued-pre-training-%E8%B7%9F-%E5%BE%AE%E8%AA%BFfine-tuning-%E4%B8%8D%E4%B8%80%E6%A8%A3-f833f9809a5c

model = FastLanguageModel.get_peft_model(
    model,
    r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    "embed_tokens", "lm_head",],  # 繼續預訓練需要更新的部分
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True,
)

from datasets import load_dataset

# 載入繁體中文資料集
dataset = load_dataset("wikicorpus", "zh", split="train")

# # 配置訓練參數
# trainer = UnslothTrainer(
#     model=model,
#     tokenizer=tokenizer,
#     train_dataset=dataset,
#     dataset_text_field="text",
#     max_seq_length=max_seq_length,
#     dataset_num_proc=8,

#     args=UnslothTrainingArguments(
#         per_device_train_batch_size=2,
#         gradient_accumulation_steps=8,
#         warmup_ratio=0.1,
#         max_steps=2000,

#         # 確保嵌入矩陣的學習率較低
#         learning_rate=5e-6,
#         embedding_learning_rate=1e-6,

#         fp16=not is_bfloat16_supported(),
#         bf16=is_bfloat16_supported(),
#         logging_steps=10,
#         save_steps=100,
#         save_total_limit=10,
#         optim="adamw_8bit",
#         weight_decay=0.00,
#         lr_scheduler_type="cosine",
#         seed=3407,
#         output_dir="./Llama-3-8B-chinese-cont-pretrain",
#     ),
# )




## CHATGPT
# # 套用 LoRA 架構，建立 LoRA 層，只訓練 0.1% ~ 0.3% 的參數，非常節省顯卡資源
# model = FastLanguageModel.get_peft_model(
#     model,c
#     r = 8,
#     lora_alpha = 16,
#     lora_dropout = 0.05,
#     bias = "none"
# )