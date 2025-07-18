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
    model_name = "TinyLlama/TinyLlama_v1.1_Chinese",
    # model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # model_name = "TinyLlama/TinyLlama-1.1B-Chat-V0.4",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)


from transformers import TextStreamer

prompt = "You are a robot"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# 設定 streamer 可以即時輸出文字
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    streamer=streamer
)

print(tokenizer.decode(inputs["input_ids"][0]))