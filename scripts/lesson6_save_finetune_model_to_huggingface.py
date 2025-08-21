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
import torch # 確保導入 torch
from datasets import load_dataset
from unsloth import UnslothTrainer, UnslothTrainingArguments
from transformers.integrations import TensorBoardCallback
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import os
from huggingface_hub import login
from peft import PeftModel # 導入 PeftModel 以便在需要時加載 adapter

max_seq_length = 2048
base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v0.4"
# 定義最終儲存和上傳的完整模型路徑和名稱
# 這是你微調後，合併了LoRA權重的完整模型的本地目錄名和Hub上的repo名
final_model_save_path = "./TinyLlama-finetune-hermes"
huggingface_repo_id = "st40404/TinyLlama-finetune-hermes" # 建議使用更具描述性的名稱

# ----------------------------------------------------------------------------------------------------
## 1. 載入模型和設定 LoRA
# ----------------------------------------------------------------------------------------------------

# 下載 & 載入模型 (用 Unsloth 最佳化版本)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model_id,
    max_seq_length = max_seq_length,
    dtype = None, # 讓 Unsloth 自動選擇最佳 dtype (通常是 bfloat16 或 float16)
    load_in_4bit = True,
)

# 使用 Unsloth 設定 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                    # 慎重考慮是否包含 embed_tokens 和 lm_head
                    # 包含它們會增加 VRAM 使用和訓練時間，但對於某些任務可能有利
                    # 如果不是做 '繼續預訓練'，通常不包含它們
                    # "embed_tokens", "lm_head",
                   ],
    lora_alpha=32,
    lora_dropout=0, # Unsloth 建議為 0
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=True, # 啟用 Unsloth 的 RSLora
)

# ----------------------------------------------------------------------------------------------------
## 2. 準備數據集
# ----------------------------------------------------------------------------------------------------

# 載入資料集
dataset = load_dataset("NousResearch/Hermes-3-Dataset", split="train")

# 格式化對話
# Unsloth 針對 Llama 3 格式進行了優化。對於 TinyLlama-Chat-V0.4，你可能需要調整。
# 這裡我假設 Hermes-3-Dataset 的 'conversations' 適合 Llama-2-Chat 或類似格式。
# TinyLlama-1.1B-Chat-V0.4 如果沒有官方明確的 Chat Template，可以嘗試 Llama-2-Chat 或 Alpaca 格式
# 如果是 Hermes-3-Dataset，它通常是 JSON 格式的 message list
def format_conversation(example):
    # Llama-2-Chat 格式範例：
    # <s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\nWhat is your favorite condiment? [/INST]
    # 如果你的數據集是這種多輪對話，需要更複雜的處理
    # 對於單輪的 instruct 數據集，可以簡化
    
    # 這裡的 'conversations' 是列表，每個元素是字典 {'from': role, 'value': content}
    messages = example["conversations"]
    formatted_text = ""
    for msg in messages:
        role = msg["from"]
        content = msg["value"]
        if role == "system":
            # TinyLlama-Chat-V0.4 不一定支援 <|system|> 標籤，可能需要調整
            formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "human":
            formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "gpt":
            formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    # 對於訓練，通常會在最後加上 <|im_end|>，但在某些情況下，如果 assistant 內容是最後一輪，可以省略
    # 確保最終輸出格式符合模型預期，否則模型會表現異常
    example["formatted_text"] = formatted_text
    return example

dataset = dataset.map(format_conversation, num_proc=os.cpu_count() or 1) # 使用所有可用的 CPU 核心

# ----------------------------------------------------------------------------------------------------
## 3. 設定訓練參數與 Trainer
# ----------------------------------------------------------------------------------------------------

# 每次 Training 時都會儲存 log
class TrainingLoggerCallback(TrainerCallback):
    def __init__(self, log_path=f"./logs/{huggingface_repo_id.split('/')[-1]}.txt"):
        self.log_path = log_path
        # 確保日誌目錄存在
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        # 清空舊檔案
        with open(self.log_path, "w") as f:
            f.write("step\tloss\tlearning_rate\n")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        loss = logs.get("loss")
        lr = logs.get("learning_rate")

        with open(self.log_path, "a") as f:
            loss_str = f"{loss:.6f}" if loss is not None else "N/A"
            lr_str = f"{lr:.8f}" if lr is not None else "N/A"
            f.write(f"{step}\t{loss_str}\t{lr_str}\n")


# 配置訓練參數
trainer = UnslothTrainer( # 注意：這裡通常是 SFTTrainer，而不是 UnslothTrainer。Unsloth 會在內部打補丁。
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="formatted_text",
    max_seq_length=max_seq_length,
    dataset_num_proc=os.cpu_count() or 1, # 使用所有可用的 CPU 核心

    args=UnslothTrainingArguments( # 注意：這裡通常是 TrainingArguments，而不是 UnslothTrainingArguments。Unsloth 會在內部打補丁。
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        max_steps=2000, # 使用 max_steps 通常比 num_train_epochs 更好，因為數據集可能很大

        # 確保嵌入矩陣的學習率較低 (僅當 'embed_tokens' 在 target_modules 中時有效)
        # 如果沒有，這個參數會被忽略。
        # learning_rate=5e-6,
        # embedding_learning_rate=1e-6, # 如果不微調 embed_tokens，請移除此行或註釋掉

        # 針對 LoRA 的典型學習率通常在 2e-4 到 5e-5 之間。5e-6 對於純粹的 LoRA 可能過低。
        learning_rate=2e-4, # 建議將此設置為更典型的 LoRA 學習率

        fp16=not torch.cuda.is_bf16_supported(), # 確保根據 GPU 支援情況使用 FP16 或 BF16
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        save_steps=100,
        save_total_limit=10,
        optim="adamw_8bit",
        weight_decay=0.00,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=final_model_save_path, # 將訓練檢查點保存到最終模型路徑
        # TensorBoard 相關設定
        report_to="tensorboard",
        logging_dir="./logs",
    ),

    callbacks=[TrainingLoggerCallback()],
)

# ----------------------------------------------------------------------------------------------------
## 4. 開始訓練
# ----------------------------------------------------------------------------------------------------
print("Starting training...")
trainer_stats = trainer.train()
print("Training finished.")

# ----------------------------------------------------------------------------------------------------
## 5. 合併 LoRA 權重並保存完整的微調模型
# ----------------------------------------------------------------------------------------------------

print("Merging LoRA weights back into the base model...")
# **關鍵：在這裡呼叫 merge_and_unload()，將 LoRA 權重合併到原始模型中**
model.merge_and_unload()
print("LoRA weights merged.")

# 保存完整的微調模型 (現在它已經包含了 LoRA 的更新)
# 保存到你定義的最終模型路徑
print(f"Saving merged model to {final_model_save_path}...")
model.save_pretrained(final_model_save_path, safe_serialization=True)
tokenizer.save_pretrained(final_model_save_path) # tokenizer 也應該與合併模型一起保存
print("Merged model and tokenizer saved locally.")

# ----------------------------------------------------------------------------------------------------
## 6. (可選) 重新載入合併後的模型進行驗證
# ----------------------------------------------------------------------------------------------------

# 這一步是可選的，主要用於驗證你保存的模型是否能正確載入。
# 你可以從本地路徑載入你剛才保存的完整模型，而不是再次加載基礎模型再套用 LoRA
from transformers import AutoModelForCausalLM, AutoTokenizer
print(f"Verifying merged model by reloading from {final_model_save_path}...")
reloaded_model = AutoModelForCausalLM.from_pretrained(
    final_model_save_path,
    trust_remote_code=True,
    # 這裡可以根據需要指定 dtype，例如 torch.bfloat16
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
).to("cuda") # 將模型移動到 GPU
reloaded_tokenizer = AutoTokenizer.from_pretrained(final_model_save_path)
print("Merged model reloaded successfully.")

# ----------------------------------------------------------------------------------------------------
## 7. 將模型和 Tokenizer 上傳到 Hugging Face Hub
# ----------------------------------------------------------------------------------------------------

# 確保已登入 Hugging Face Hub
# 如果您在終端機中運行 `huggingface-cli login`，則無需在代碼中再次登入
# login() # 僅在您尚未登入時解除註釋並運行一次

print(f"Pushing merged model to Hugging Face Hub: {huggingface_repo_id}...")
# 將 "reloaded_model" (即已經合併的完整模型) 推送到 Hub
reloaded_model.push_to_hub(huggingface_repo_id, private=True)
reloaded_tokenizer.push_to_hub(huggingface_repo_id, private=True)
print("Merged model and tokenizer pushed to Hugging Face Hub successfully!")# Citation of TinyLlama
