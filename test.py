import torch.nn as nn
import torch
import gensim
from khaiii import KhaiiiApi

def embedding(word):
    model = gensim.models.Word2Vec.load('/mnt/e/Temp/web/model_pth/mymodel_0521')
    word = model.wv[word]
    return word

def sentence_enbedding(sentence):
    api = KhaiiiApi()
    tokenized_sentence = []
    for word in api.analyze(sentence):
        for morph in word.morphs:
            tokenized_sentence.append(morph.lex)
    word_lst = []
    for token in tokenized_sentence:
        word_lst.append(embedding(token))
    return word_lst

word = "안녕"
emb = embedding(word)
print(emb)
print(emb.shape)
a = emb.shape
dim = a[0]
print(dim)
