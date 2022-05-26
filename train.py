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

def preprocess(n):

    print("reading sntences start")
    Q, A = get_chat_data(n)
    pairs = [[Q[i],A[i]] for i in range(n)]
    
    source_vocab = Vocab()
    target_vocab = Vocab()

    for pair in pairs:
        source_vocab.add_vocab(pair[0])
        target_vocab.add_vocab(pair[1])

    print("source vocab size =", source_vocab.n_vocab)
    print("target vocab size =", target_vocab.n_vocab)

    return pairs, source_vocab, target_vocab

def train(pairs, encoder, decoder, n_iter, print_every=1000, learning_rate=0.01):
    loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)


    criterion = nn.NLLLoss()

    for i in tqdm(range(1, n_iter + 1)):
        Q = [pairs[i-1][0]]
        A = [pairs[i-1][1]]
        print(Q, A)
        encoder_hidden = torch.zeros([1, 1, encoder.hidden_size]).to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss = 0


        source_length = 0


        for enc_input in range(source_length):
            _, encoder_hidden = encoder(source_tensor[enc_input], encoder_hidden)

        decoder_input = torch.Tensor([[SOS_token]]).long().to(device)
        decoder_hidden = encoder_hidden # connect encoder output to decoder input

        for di in range(target_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # teacher forcing

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        loss_iter = loss.item() / target_length
        loss_total += loss_iter

        if i % print_every == 0:
            loss_avg = loss_total / print_every
            loss_total = 0
            print("[{} - {}%] loss = {:05.4f}".format(i, i / n_iter * 100, loss_avg))

def evaluate(pairs, source_vocab, target_vocab, encoder, decoder, target_max_length):
    for pair in pairs:
        print(">", pair[0])
        print("=", pair[1])
        source_tensor = tensorize(source_vocab, pair[0])
        source_length = source_tensor.size()[0]
        encoder_hidden = torch.zeros([1, 1, encoder.hidden_size]).to(device)

        for ei in range(source_length):
            _, encoder_hidden = encoder(source_tensor[ei], encoder_hidden)

        decoder_input = torch.Tensor([[SOS_token]], device=device).long()
        decoder_hidden = encoder_hidden
        decoded_words = []

        for di in range(target_max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            _, top_index = decoder_output.data.topk(1)
            if top_index.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(target_vocab.index2vocab[top_index.item()])

            decoder_input = top_index.squeeze().detach()

        predict_words = decoded_words
        predict_sentence = " ".join(predict_words)
        print("<", predict_sentence)
        print("")
'''
SOURCE_MAX_LENGTH = 100
TARGET_MAX_LENGTH = 10000

load_pairs, load_source_vocab, load_target_vocab = preprocess(6000)
print(random.choice(load_pairs))

enc_hidden_size = 16
dec_hidden_size = enc_hidden_size

enc = Encoder(load_source_vocab.n_vocab, enc_hidden_size).to(device)
dec = Decoder(dec_hidden_size, load_target_vocab.n_vocab).to(device)

train(load_pairs, load_source_vocab, load_target_vocab, enc, dec, 5000, print_every=1000)

evaluate(load_pairs, load_source_vocab, load_target_vocab, enc, dec, TARGET_MAX_LENGTH)
'''

Q, A = get_chat_data(100)
Q_input = make_all_data(Q)
print(Q_input)
