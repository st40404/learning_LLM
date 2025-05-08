#######################################################################################
##  reference from : https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html ##
#######################################################################################

import torch
from transformers import BertTokenizer
from IPython.display import clear_output

PRETRAINED_MODEL_NAME = "bert-base-chinese"  # 指定繁簡中文 BERT-BASE 預訓練模型

# 取得此預訓練模型所使用的 tokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

clear_output()

#####################################
## lesson 1 : show pytorch version ##
#####################################
# print("PyTorch 版本：", torch.__version__)

###################################################
## lesson 2 : show total amount of chinese group ##
###################################################
# vocab = tokenizer.vocab
# print("字典大小：", len(vocab))

##########################################################
## lesson 3 : show specific word group of chinese group ##
##########################################################
# import random
# vocab = tokenizer.vocab
# random_tokens = random.sample(list(vocab), 10)
# random_ids = [vocab[t] for t in random_tokens]

# print("{0:20}{1:15}".format("token", "index"))
# print("-" * 25)
# for t, id in zip(random_tokens, random_ids):
#     print("{0:15}{1:10}".format(t, id))

#########################################################################################
## lesson 4 : get correspond character of the sentence, aslo introduce special tokens  ##
#########################################################################################
## [CLS] (classification) ：在做分類任務時其最後一層的 repr. 會被視為整個輸入序列的 repr.
## [SEP] (Separator)      ：有兩個句子的文本會被串接成一個輸入序列，並在兩句之間插入這個 token 以做區隔
## [UNK] (Unknown)        ：沒出現在 BERT 字典裡頭的字會被這個 token 取代
## [PAD] (Padding)        ：zero padding 遮罩，將長度不一的輸入序列補齊方便做 batch 運算
## [MASK]                 ：未知遮罩，僅在預訓練階段會用到

# text = "[CLS] 等到潮水 [MASK] 了，就知道誰沒穿褲子。"
# tokens = tokenizer.tokenize(text)
# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(text)
# print(tokens[:17], '...')
# print(ids[:17], '...')

######################################################
## lesson 5 : fullfill MASK token by trained model  ##
######################################################
# from transformers import BertForMaskedLM

# text = "[CLS] 等到潮水 [MASK] 了，就知道誰沒穿褲子。"
# tokens = tokenizer.tokenize(text)
# ids = tokenizer.convert_tokens_to_ids(tokens)

# # 除了 tokens 以外我們還需要辨別句子的 segment ids
# tokens_tensor = torch.tensor([ids])  # (1, seq_len)
# segments_tensors = torch.zeros_like(tokens_tensor)  # (1, seq_len)
# maskedLM_model = BertForMaskedLM.from_pretrained(PRETRAINED_MODEL_NAME)
# clear_output()

# # 使用 masked LM 估計 [MASK] 位置所代表的實際 token 
# maskedLM_model.eval()
# with torch.no_grad():
#     outputs = maskedLM_model(tokens_tensor, segments_tensors)
#     predictions = outputs[0]
#     # (1, seq_len, num_hidden_units)
# del maskedLM_model

# # 將 [MASK] 位置的機率分佈取 top k 最有可能的 tokens 出來
# masked_index = 5
# k = 3
# probs, indices = torch.topk(torch.softmax(predictions[0, masked_index], -1), k)
# predicted_tokens = tokenizer.convert_ids_to_tokens(indices.tolist())

# # 顯示 top k 可能的字。一般我們就是取 top 1 當作預測值
# print("輸入 tokens :", tokens[:10], '...')
# print('-' * 50)
# for i, (t, p) in enumerate(zip(predicted_tokens, probs), 1):
#     tokens[masked_index] = t
#     print("Top {} ({:2}%):{}".format(i, int(p.item() * 100), tokens[:10]), '...')

#########################################################
## lesson 6 : tune reconize related or unrelated info  ##
#########################################################
## get date from : https://www.kaggle.com/c/fake-news-pair-classification-challenge/data

# import pandas as pd

# # 簡單的數據清理，去除空白標題的 examples
# df_train = pd.read_csv("./dataset/BERT/lesson6/fake-news-pair-classification-challenge/train.csv")
# empty_title = ((df_train['title2_zh'].isnull()) \
#                | (df_train['title1_zh'].isnull()) \
#                | (df_train['title2_zh'] == '') \
#                | (df_train['title2_zh'] == '0'))
# df_train = df_train[~empty_title]

# # 剔除過長的樣本以避免 BERT 無法將整個輸入序列放入記憶體不多的 GPU
# MAX_LENGTH = 30
# df_train = df_train[~(df_train.title1_zh.apply(lambda x : len(x)) > MAX_LENGTH)]
# df_train = df_train[~(df_train.title2_zh.apply(lambda x : len(x)) > MAX_LENGTH)]

# # 只用 1% 訓練數據看看 BERT 對少量標註數據有多少幫助
# SAMPLE_FRAC = 0.01
# df_train = df_train.sample(frac=SAMPLE_FRAC, random_state=9527)

# # 去除不必要的欄位並重新命名兩標題的欄位名
# df_train = df_train.reset_index()
# df_train = df_train.loc[:, ['title1_zh', 'title2_zh', 'label']]
# df_train.columns = ['text_a', 'text_b', 'label']

# # idempotence, 將處理結果另存成 tsv 供 PyTorch 使用
# df_train.to_csv("train.tsv", sep="\t", index=False)

# print("訓練樣本數：", len(df_train))
# df_train.head()


#########################################################
## lesson 7: transfer original article to BERT format  ##
#########################################################
# from torch.utils.data import Dataset
# import pandas as pd
# import pysnooper

# class FakeNewsDataset(Dataset):
#     # 讀取前處理後的 tsv 檔並初始化一些參數
#     def __init__(self, mode, tokenizer):
#         assert mode in ["train", "test"]  # 一般訓練你會需要 dev set
#         self.mode = mode
#         # 大數據你會需要用 iterator=True
#         self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
#         self.len = len(self.df)
#         self.label_map = {'agreed': 0, 'disagreed': 1, 'unrelated': 2}
#         self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer

#     @pysnooper.snoop() # 加入以了解所有轉換過程
#     # 定義回傳一筆訓練 / 測試數據的函式
#     def __getitem__(self, idx):
#         if self.mode == "test":
#             text_a, text_b = self.df.iloc[idx, :2].values
#             label_tensor = None
#         else:
#             text_a, text_b, label = self.df.iloc[idx, :].values
#             # 將 label 文字也轉換成索引方便轉換成 tensor
#             label_id = self.label_map[label]
#             label_tensor = torch.tensor(label_id)

#         # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
#         word_pieces = ["[CLS]"]
#         tokens_a = self.tokenizer.tokenize(text_a)
#         word_pieces += tokens_a + ["[SEP]"]
#         len_a = len(word_pieces)

#         # 第二個句子的 BERT tokens
#         tokens_b = self.tokenizer.tokenize(text_b)
#         word_pieces += tokens_b + ["[SEP]"]
#         len_b = len(word_pieces) - len_a

#         # 將整個 token 序列轉換成索引序列
#         ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
#         tokens_tensor = torch.tensor(ids)
        
#         # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
#         segments_tensor = torch.tensor([0] * len_a + [1] * len_b, 
#                                         dtype=torch.long)

#         return (tokens_tensor, segments_tensor, label_tensor)
    
#     def __len__(self):
#         return self.len


# # 初始化一個專門讀取訓練樣本的 Dataset，使用中文 BERT 斷詞
# trainset = FakeNewsDataset("train", tokenizer=tokenizer)

# # 選擇第一個樣本
# sample_idx = 1

# # 將原始文本拿出做比較
# text_a, text_b, label = trainset.df.iloc[sample_idx].values

# # 利用剛剛建立的 Dataset 取出轉換後的 id tensors
# tokens_tensor, segments_tensor, label_tensor = trainset[sample_idx]

# # 將 tokens_tensor 還原成文本
# tokens = tokenizer.convert_ids_to_tokens(tokens_tensor.tolist())
# combined_text = "".join(tokens)

# # 渲染前後差異，毫無反應就是個 print。可以直接看輸出結果
# print(f"""[原始文本]
# 句子 1：{text_a}
# 句子 2：{text_b}
# 分類  ：{label}

# --------------------

# [Dataset 回傳的 tensors]
# tokens_tensor  ：{tokens_tensor}

# segments_tensor：{segments_tensor}

# label_tensor   ：{label_tensor}

# --------------------

# [還原 tokens_tensors]
# {combined_text}
# """)



#########################################################
## lesson 8:   ##
#########################################################
"""
實作可以一次回傳一個 mini-batch 的 DataLoader
這個 DataLoader 吃我們上面定義的 `FakeNewsDataset`，
回傳訓練 BERT 時會需要的 4 個 tensors：
- tokens_tensors  : (batch_size, max_seq_len_in_batch)
- segments_tensors: (batch_size, max_seq_len_in_batch)
- masks_tensors   : (batch_size, max_seq_len_in_batch)
- label_ids       : (batch_size)
"""

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import pandas as pd

class FakeNewsDataset(Dataset):
    # 讀取前處理後的 tsv 檔並初始化一些參數
    def __init__(self, mode, tokenizer):
        assert mode in ["train", "test"]  # 一般訓練你會需要 dev set
        self.mode = mode
        # 大數據你會需要用 iterator=True
        self.df = pd.read_csv(mode + ".tsv", sep="\t").fillna("")
        self.len = len(self.df)
        self.label_map = {'agreed': 0, 'disagreed': 1, 'unrelated': 2}
        self.tokenizer = tokenizer  # 我們將使用 BERT tokenizer

    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):
        if self.mode == "test":
            text_a, text_b = self.df.iloc[idx, :2].values
            label_tensor = None
        else:
            text_a, text_b, label = self.df.iloc[idx, :].values
            # 將 label 文字也轉換成索引方便轉換成 tensor
            label_id = self.label_map[label]
            label_tensor = torch.tensor(label_id)

        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces += tokens_a + ["[SEP]"]
        len_a = len(word_pieces)

        # 第二個句子的 BERT tokens
        tokens_b = self.tokenizer.tokenize(text_b)
        word_pieces += tokens_b + ["[SEP]"]
        len_b = len(word_pieces) - len_a

        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0，其他為 1 表示第二句
        segments_tensor = torch.tensor([0] * len_a + [1] * len_b, 
                                        dtype=torch.long)

        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len


# 初始化一個專門讀取訓練樣本的 Dataset，使用中文 BERT 斷詞
trainset = FakeNewsDataset("train", tokenizer=tokenizer)

# 這個函式的輸入 `samples` 是一個 list，裡頭的每個 element 都是
# 剛剛定義的 `FakeNewsDataset` 回傳的一個樣本，每個樣本都包含 3 tensors：
# - tokens_tensor
# - segments_tensor
# - label_tensor
# 它會對前兩個 tensors 作 zero padding，並產生前面說明過的 masks_tensors
def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    # 測試集有 labels
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, 
                                    batch_first=True)
    
    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids


# 初始化一個每次回傳 64 個訓練樣本的 DataLoader
# 利用 `collate_fn` 將 list of samples 合併成一個 mini-batch 是關鍵
BATCH_SIZE = 64
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, 
                         collate_fn=create_mini_batch)

data = next(iter(trainloader))

tokens_tensors, segments_tensors, \
    masks_tensors, label_ids = data

print(f"""
tokens_tensors.shape   = {tokens_tensors.shape} 
{tokens_tensors}
------------------------
segments_tensors.shape = {segments_tensors.shape}
{segments_tensors}
------------------------
masks_tensors.shape    = {masks_tensors.shape}
{masks_tensors}
------------------------
label_ids.shape        = {label_ids.shape}
{label_ids}
""")