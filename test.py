import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings

import simplechinese as sc

pinyin = sc.str2pinyin("【确认题示】因您长时间观看堷短视蘋，栯噫愿可幇短视蘋婰赞，ㄖ給55-388咪，+\/：15134432717", hasTone=False)
print(pinyin)