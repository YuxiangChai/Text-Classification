import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, vocab_size, embed_dim, output_dim, window_sizes=(1, 2, 3, 5)):
        super(CNN, self).__init__()
        self.embed = nn.Embedding(vocab_size+1, embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, 20, [window_size, embed_dim], padding=(window_size-1, 0))
                                    for window_size in window_sizes])
        self.fc = nn.Linear(20*len(window_sizes), output_dim)

    def forward(self, x):
        x = self.embed(x)
        x = torch.unsqueeze(x, 1)
        xs = []
        for conv in self.convs:
            x2 = torch.tanh(conv(x))
            x2 = torch.squeeze(x2, -1)
            x2 = F.max_pool1d(x2, x2.size(2))
            xs.append(x2)
        x = torch.cat(xs, 2)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class FastText(nn.Module):

    def __init__(self, vocab_size, embed_dim, output_dim):
        super(FastText, self).__init__()
        self.embed = nn.Embedding(vocab_size+1, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.embed(x)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class LSTM(nn.Module):

    def __init__(self, vocab_size, embed_dim, output_dim):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size+1, embed_dim)
        self.lstm = nn.LSTM(embed_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x, l):
        x = self.embed(x)
        lstm_out, (h, c) = self.lstm(x)
        x = self.fc(h[-1])
        return x
