import torch.nn as nn
import torch
import torch.nn.functional as F

import gensim

class EncoderRNN(nn.Module):
    def __init__(self, word, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.word2vec = gensim.models.Word2Vec.load('/mnt/e/Temp/web/model_pth/mymodel_0521')
        self.embedding = self.word2vec.wv[word]
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, word):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.word2vec = gensim.models.Word2Vec.load('/mnt/e/Temp/web/model_pth/mymodel_0522')
        self.embedding = self.word2vec.wv[word]
        
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

