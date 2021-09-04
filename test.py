import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# model = BertModel.from_pretrained('bert-base-chinese')
# print(model.embeddings.word_embeddings)
# print(tokenizer.vocab['[PAD]'])
# print(len(tokenizer.vocab))
# print(BertConfig.from_json_file("bert-base-chinese/bert_config.json"))