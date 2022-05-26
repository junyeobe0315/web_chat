from khaiii_utils import *
import json
from tqdm import tqdm
from gensim.models import Word2Vec
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

def make_only_kr(q,a):
    Q = []
    A = []
    for sentence in q:
        sentence = sentence.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
        Q.append(sentence)
    for sentence in a:
        sentence = sentence.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
        A.append(sentence)
    return Q, A

def tokenize_sentence(spell_sentence):
    '''
    문장을 토큰화.
    input 문장
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
    Q, A = make_only_kr(q, a)

    sentences = list()
    
    for sentence in tqdm(Q):
        temp = tokenize_sentence(sentence)
        sentences.append(temp)
    for sentence in tqdm(A):
        temp = tokenize_sentence(sentence)
        sentences.append(temp)
    
    print("Word2vec training start")
    model = Word2Vec(sentences, window=100, sg=10, min_count=1)
    print("traning end")

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


if __name__ == "__main__":
    n = 6000

    model = train_word2vec(n)
    print('word2vec embedding vector shape :', model.wv.vectors.shape)

    model.save('/mnt/e/Temp/web/model_pth/mymodel_0526')