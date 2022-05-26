import numpy as np
import random
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim

from tqdm import tqdm

import gensim

from khaiii_utils import *
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def word_embedding(word):
    model = gensim.models.Word2Vec.load('/mnt/e/Temp/web/model_pth/mymodel_0521')
    print(model.wv[word])
    return model

def sentence_to_index(sentence):
    sentence = make_no_space(sentence)
    sentence = spacing(sentence)
    sentence = spell_check(sentence)
    sentence = tokenize_word(sentence)
    return sentence
    
def get_chat_data(n):
    '''
    json 파일에서 질문과 답변을 가져옴.
    q,a 는 리스트 형식으로 저장함.
    input은 불러올 데이터 수
    output은 q,a 리스트
    '''
    with open('ko_wiki_v1_squad.json', 'r') as f:
        data = json.load(f)
    q = []
    a = []
    for i in range(n):
        try:
            q.append(data["data"][i]["paragraphs"][0]["qas"][0]["question"])
            a.append(data["data"][i]["paragraphs"][0]["context"])
        except:
            pass
    return q, a



SOS_token = 0
EOS_token = 1


class Chat:
    def __init__(self, name):
        self.name = name
        self.word2index = {} # 워드에 번호 매기기
        self.word2count = {} # 어떤 워드들 들어있는지 얼마나 들어있는지
        self.index2word = {0: "SOS", 1: "EOS"} # 2 : "word"
        self.n_words = 2  # SOS 와 EOS 포함

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
    
    def addSentence(self, sentence):
        api = KhaiiiApi()
        for word in api.analyze(sentence):
            self.addWord(word)

# -------------------------두 가지 버젼 실험 ---------------------------

class Vocab:
    def __init__(self):
        self.vocab2index = {"<SOS>": SOS_token, "<EOS>": EOS_token}
        self.index2vocab = {SOS_token: "<SOS>", EOS_token: "<EOS>"}
        self.vocab_count = {}
        self.n_vocab = len(self.vocab2index)

    def add_vocab(self, sentence):
        api = KhaiiiApi()
        for word in api.analyze(sentence):
            if word not in self.vocab2index:
                self.vocab2index[word] = self.n_vocab
                self.vocab_count[word] = 1
                self.index2vocab[self.n_vocab] = word
                self.n_vocab += 1
            else:
                self.vocab_count[word] += 1

# --------------------------------------------------------------------------

def readChats(n, reverse=False):
    print("Reading data...")

    # 질문과 답변 불러오기
    Q, A = get_chat_data(n)

    # Q, A pairs 만들기
    pairs = [[Q[i],A[i]] for i in range(len(n))]

    # 쌍을 뒤집고, Lang 인스턴스 생성
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_data = Chat(Q) # input data 에 Q 데이터 집어넣음
        output_data = Chat(A) # output data 에 A 데이터 집어넣음

    else:
        input_data = Chat(Q)
        output_data = Chat(A)

    return input_data, output_data, pairs

def preprocess()


def prepareData(reverse=False):
    input_data, output_data, pairs = readChats(reverse)
    print("number of pairs :", len(pairs))
    for pair in tqdm(pairs):
        input_data.addSentence(pair[0])
        output_data.addSentence(pair[1])

    return input_data, output_data, pairs

input_data, output_data, pairs = prepareData(False)
print(random.choice(pairs))

