from unsloth import FastLanguageModel
# Unsloth 會自動呼叫 Hugging Face API 幫你抓下整個模型並快取在本機的 ~/.cache/huggingface 路徑
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-4-Scout-17B-16E",
    max_seq_length = 4096,
    dtype = None,                                   # "auto" for BF16, float16, etc.
    load_in_4bit = True,                            # or False if GPU VRAM 夠
    token = "your_token" # model token
)