import numpy as np
import random
import json
import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # SOS 와 EOS 포함

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def readChats(question, answer, reverse=False):
    print("Reading data...")

    # 질문과 답변 불러오기
    Q, A = get_chat_data(6000)

    # Q, A pairs 만들기
    pairs = [[Q[i],A[i]] for i in range(len(Q))]

    # 쌍을 뒤집고, Lang 인스턴스 생성
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_data = Chat(question)
        output_data = Chat(answer)
    else:
        input_data = Chat(question)
        output_data = Chat(answer)

    return input_data, output_data, pairs

def prepareData(Q, A, reverse=False):
    input_data, output_data, pairs = readChats(Q, A, reverse)
    print("number of pairs :", len(pairs))
    for pair in tqdm(pairs):
        input_data.addSentence(pair[0])
        output_data.addSentence(pair[1])

    return input_data, output_data, pairs

Q, A = get_chat_data(6000)

input_data, output_data, pairs = prepareData(Q, A, False)
print(random.choice(pairs))

sentence = "서부히스밭쥐가 주로 서식하는 지형조건은"

def tensorFromSentence(sentence):
    api = KhaiiiApi()
    for word in api.analyze(sentence):
        for morph in word.morphs:
            indexes = word_embedding(morph)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(pair[0])
    target_tensor = tensorFromSentence(pair[1])
    return (input_tensor, target_tensor)

teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=10000):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing 포함: 목표를 다음 입력으로 전달
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Teacher forcing 미포함: 자신의 예측을 다음 입력으로 사용
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # 입력으로 사용할 부분을 히스토리에서 분리

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # print_every 마다 초기화
    plot_loss_total = 0  # plot_every 마다 초기화

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in tqdm(range(1, n_iters + 1)):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # 주기적인 간격에 이 locator가 tick을 설정
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

def evaluate(encoder, decoder, sentence, max_length=100000):
    with torch.no_grad():
        input_tensor = tensorFromSentence(sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in tqdm(range(max_length)):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_data.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, n=10):
    for i in tqdm(range(n)):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
    
hidden_size = 256
encoder1 = EncoderRNN(input_data.n_words, hidden_size).to(device)
decoder1 = DecoderRNN(hidden_size, output_data.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, decoder1, 75000, print_every=5000)
evaluateRandomly(encoder1, decoder1)