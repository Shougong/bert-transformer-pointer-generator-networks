import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig

def data_augment(string):
    pass

text = '【确认题示】因您长时间观看音短视苹，栯噫愿可帮短视苹婰赞，日给55-388咪，+\/：15134432717'
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
tokens = tokenizer.tokenize(text)
print(tokens)