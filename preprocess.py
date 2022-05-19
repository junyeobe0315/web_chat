from khaiii_utils import *
import json
from tqdm import tqdm
from gensim.models.word2vec import Word2Vec
import numpy as np

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

def tokenize_word(spell_sentence):
    '''
    문장을 토큰화.
    input 문장 float
    output 토큰화 된 리스트
    '''
    api = KhaiiiApi()
    tokenized_sentence = []
    for word in api.analyze(spell_sentence):
        for morph in word.morphs:
            tokenized_sentence.append(morph.lex)
    return tokenized_sentence

def train_word2vec(n, acc=True):
    q, a = get_chat_data(n)
    sentences = list()
    
    for sentence in tqdm(q):
        temp = tokenize_word(sentence)
        sentences.append(temp)
    for sentence in tqdm(a):
        temp = tokenize_word(sentence)
        sentences.append(temp)
    model = Word2Vec(sentences, window=15)
    model.init_sims(replace=True)
    if acc:
        while True:
            word = input("유사단어 찾기(quit to exit) :")
            if word == "quit":
                break
            else:
                try:
                    print(model.wv.most_similar(word))
                except:
                    print("{} is not in model".format(word))
    return model

n = 6000

model = train_word2vec(n)

