import torch.nn as nn
import torch


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classess):
        super(NeuralNet, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_3 = nn.Linear(hidden_size, num_classess)
        self.relu = nn.ReLU()

    def forward(self, x):
        out_1 = self.layer_1(x)
        out_1_ = self.relu(out_1)
        out_2 = self.layer_2(out_1_)
        out_2_ = self.relu(out_2)
        out_3 = self.layer_3(out_2_)
        return out_3


class LSTM(nn.module):
    def __init__(self):
        super(TextLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=n_size, hidden_size=n_hidden, dropout=0.3)
        self.W = nn.Parameter(torch.randn([n_hidden, n_size]).type(dtype))
        self.b = nn.Parameter(torch.randn([n_size]).type(dtype))
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, hidden_and_cell, X):
        X = X.transpose(0, 1)
        outputs, hidden = self.lstm(X, hidden_and_cell)
        outputs = outputs[-1]
        model = torch.mm(outputs, self.W) + self.b
        return model
        
