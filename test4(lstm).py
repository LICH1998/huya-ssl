import torch
import warnings
warnings.filterwarnings('ignore')  # 警告扰人，手动封存
import gensim
torch.manual_seed(2)
datas=[('你 叫 什么 名字 ?','n v n n f'),('今天 天气 怎么样 ?','n n adj f'),]
words=[ data[0].split() for data in datas]
tags=[ data[1].split() for data in datas]
print(words)
id2word=gensim.corpora.Dictionary(words)
word2id=id2word.token2id
print(id2word)