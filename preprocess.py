from khaiii_utils import *
import json
from tqdm import tqdm
import gensim
from gensim.models import Word2Vec
import numpy as np
from word2vec import get_chat_data

def word_embedding(word):
    model = gensim.models.Word2Vec.load('/mnt/e/Temp/web/model_pth/mymodel_0526')
    print(model.wv[word])
    return model.wv[word]


def tokenize_sentence(sentence):
    api = KhaiiiApi()
    tokenized_sentence = []
    for word in api.analyze(sentence):
        for morph in word.morphs:
            tokenized_sentence.append(morph.lex)
    print(tokenized_sentence)
    return tokenized_sentence

def make_input_data(tokenized_sentence):
    input_data = []
    for word in tokenized_sentence:
        print(word)
        input_data.append(word_embedding(word))
    print(input_data)
    return input_data

def make_all_data(Q):
    input_data = []
    for sentence in Q:
        print(sentence)
        word_lst = tokenize_sentence(sentence)
        input_data.append(make_input_data(word_lst))
    return input_data

Q, A = get_chat_data(3)
Q_input = make_all_data(Q)
for data in Q_input:
    print(data)
    print("------------------------------------")