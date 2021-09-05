import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig

# text = "等到潮水退去了。"

# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# vocab = tokenizer.vocab
# inv_vocab = {ind:word for word, ind in vocab.items()}
# inputs = tokenizer(text)
# print(inputs)
# print(inv_vocab[101])
# model = BertModel.from_pretrained('bert-base-chinese')
# print(model.embeddings)
# print(tokenizer.vocab['[PAD]'])
# print(len(tokenizer.vocab))
# print(BertConfig.from_json_file("bert-base-chinese/bert_config.json"))

# a = torch.scatter_(y[0], y[1], [200, 21128])