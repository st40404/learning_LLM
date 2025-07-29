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

max_seq_length = 2048

# 下載 & 載入模型 (用 Unsloth 最佳化版本) 
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

# 載入資料集 (回覆對話的類別)
dataset = load_dataset("NousResearch/Hermes-3-Dataset", split="train")

def format_conversation(example):
    messages = example["conversations"]
    formatted = ""
    for msg in messages:
        role = msg["from"]
        content = msg["value"]
        if role == "system":
            formatted += f"<|system|>\n{content}\n"
        elif role == "human":
            formatted += f"<|user|>\n{content}\n"
        elif role == "gpt":
            formatted += f"<|assistant|>\n{content}\n"
    example["formatted_text"] = formatted
    return example

dataset = dataset.map(format_conversation, num_proc=12)


from unsloth import UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported
from transformers import TrainingArguments


# 配置訓練參數
trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="formatted_text",
    max_seq_length=max_seq_length,
    dataset_num_proc=12,

    args=UnslothTrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        max_steps=2000,

        # 確保嵌入矩陣的學習率較低
        learning_rate=5e-6,
        embedding_learning_rate=1e-6,

        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        save_steps=100,
        save_total_limit=10,
        optim="adamw_8bit",
        weight_decay=0.00,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="./TinyLlama-1.1B-Chat-V0.4-pretrain",
    ),
)
# 參數介紹
"""
model : 要使用的 model
tokenizer : 對文字進行 tokenization 的 tokenizer
train_dataset : 訓練用的資料集
dataset_text_field : 資料集中, 哪個欄位是你要拿來當訓練文字用的
max_seq_length : 訓練時每個樣本的最大 token 數
dataset_num_proc : 多處理程序來加速 tokenization, 用多少 CPU 核心
"""

## 基本訓練設定
"""
per_device_train_batch_size : 每個 GPU 上的訓練批次大小。若你使用 1 張 GPU, 總 batch size 為 2
gradient_accumulation_steps : 累積 8 次小 batch 的梯度後才進行一次反向傳播, 有效 batch size 為 2 * 8 = 16。適合顯存不足時使用
max_steps : 訓練最多進行 2000 個訓練步驟 (iterations)
"""

## 學習率與優化器設定
"""
learning_rate : 預設參數的學習率。對大部分 transformer 層有效。
embedding_learning_rate : 嵌入層 (embedding layer) 的專用學習率, 通常設定較低, 以避免詞嵌入不穩定。
optim : 使用 8-bit AdamW 優化器, 可大幅減少 GPU 記憶體佔用。
weight_decay : 權重衰減 (L2正則化) 係數, 設為 0 表示不進行權重衰減。
lr_scheduler_type : 學習率調整策略, cosine 表示使用餘弦衰減 (cosine decay) 。學習率會先增加 (warmup) , 再緩慢下降。
warmup_ratio : 訓練前 10% 的步數作為 warmup, 學習率從 0 緩慢上升至目標值, 避免訓練初期發散。
"""

## 精度與硬體加速設定
"""
fp16 : 如果硬體不支援 BF16, 就使用 FP16 (半精度浮點數) 以節省記憶體與加速訓練。
bf16 : 若硬體支援 BF16 (如 A100、4090) , 則使用 BF16 訓練, 兼具穩定性與效率。兩個參數通常會對應互補。
"""

## 日誌與模型儲存
"""
logging_steps : 每訓練 10 個 step, 輸出一次訓練日誌 (如損失值) 。
save_steps : 每訓練 100 個 step 儲存一次模型 checkpoint。
save_total_limit : 最多保留 10 個 checkpoint, 超過時會自動刪除最舊的。
"""

## 其他設定
"""
seed : 設定隨機種子, 讓訓練結果可重現。
output_dir : 模型儲存的輸出資料夾。完成訓練後, 會將最佳模型與最後 checkpoint 存在這個資料夾。
"""

# 開始訓練
trainer_stats = trainer.train()
