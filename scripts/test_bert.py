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

#########################################
## lesson 6 : tune reconize fake news  ##
#########################################
## get date from : https://www.kaggle.com/c/fake-news-pair-classification-challenge/data

import pandas as pd

# 簡單的數據清理，去除空白標題的 examples
df_train = pd.read_csv("../dataset/BERT/lesson6/fake-news-pair-classification-challenge/train.csv")
empty_title = ((df_train['title2_zh'].isnull()) \
               | (df_train['title1_zh'].isnull()) \
               | (df_train['title2_zh'] == '') \
               | (df_train['title2_zh'] == '0'))
df_train = df_train[~empty_title]

# 剔除過長的樣本以避免 BERT 無法將整個輸入序列放入記憶體不多的 GPU
MAX_LENGTH = 30
df_train = df_train[~(df_train.title1_zh.apply(lambda x : len(x)) > MAX_LENGTH)]
df_train = df_train[~(df_train.title2_zh.apply(lambda x : len(x)) > MAX_LENGTH)]

# 只用 1% 訓練數據看看 BERT 對少量標註數據有多少幫助
SAMPLE_FRAC = 0.01
df_train = df_train.sample(frac=SAMPLE_FRAC, random_state=9527)

# 去除不必要的欄位並重新命名兩標題的欄位名
df_train = df_train.reset_index()
df_train = df_train.loc[:, ['title1_zh', 'title2_zh', 'label']]
df_train.columns = ['text_a', 'text_b', 'label']

# idempotence, 將處理結果另存成 tsv 供 PyTorch 使用
df_train.to_csv("train.tsv", sep="\t", index=False)

print("訓練樣本數：", len(df_train))
df_train.head()