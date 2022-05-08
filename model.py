import torch.nn as nn


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
