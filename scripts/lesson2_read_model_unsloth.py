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

# Unsloth 無法使用 local 端的 model
# Unsloth 會自動呼叫 Hugging Face API 幫你抓下整個模型並快取在本機的 ~/.cache/huggingface 路徑
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "meta-llama/Llama-4-Scout-17B-16E",
#     max_seq_length = 4096,
#     dtype = None,                                   # "auto" for BF16, float16, etc.
#     load_in_4bit = True,                            # or False if GPU VRAM 夠
#     token = "your_token" # model token
# )


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "TinyLlama/TinyLlama_v1.1",
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True
)

# 檢查是否成功
print(model)
print(tokenizer)