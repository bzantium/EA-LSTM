import torch
import torch.nn as nn
from utils import make_cuda


class weightedLSTM(nn.Module):
    def __init__(self, n_features, n_hidden, n_layers, n_output, weight, bidirectional):
        super(weightedLSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.weight = make_cuda(torch.FloatTensor(weight))
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden,
                            num_layers=n_layers, bidirectional=bidirectional,
                            batch_first=True)
        self.regr = nn.Linear(2 * n_hidden if bidirectional else n_hidden, n_output)

    def forward(self, inputs):  # inputs: [batch_size, time_steps, n_features]
        n_batch = inputs.size(0)
        inputs = torch.transpose(torch.transpose(inputs, -1, -2) * self.weight, -1, -2)
        _, (hidden, cell) = self.lstm(inputs)
        hidden = hidden.view(self.n_layers, 2 if self.bidirectional else 1, n_batch, self.n_hidden)
        if self.bidirectional:
            f_hidden, b_hidden = hidden[-1]
            hidden = torch.cat((f_hidden, b_hidden), dim=1)
        else:
            hidden = hidden[-1]
        out = self.regr(hidden)
        return out.view(-1)
