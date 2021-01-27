import torch
import torch.nn as nn


class RnnScore(nn.Module):
    def __init__(self, num_q: int):
        super(RnnScore, self).__init__()
        self.q_embedding = nn.Embedding(num_q, 32)
        self.h_layer = nn.Linear(32, 16)
        self.c_layer = nn.Linear(32, 16)

        self.cnn = nn.Sequential(nn.Conv1d(2, 64, 60, stride=6),
                                 nn.ReLU(inplace=True),
                                 nn.Conv1d(64, 32, 30, stride=3),
                                 nn.ReLU(inplace=True),
                                 nn.Conv1d(32, 16, 10))
        self.rnn = nn.LSTM(16, 16, batch_first=True)
        self.score_layer = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        q_code = self.q_embedding(q)
        h = self.h_layer(q_code).unsqueeze(0)
        c = self.c_layer(q_code).unsqueeze(0)
        features = self.cnn(x).transpose(1, 2)
        output, _ = self.rnn(features, (h, c))
        return self.score_layer(output[:, -1])


class RankNet(nn.Module):
    def __init__(self, num_q: int):
        super(RankNet, self).__init__()
        self.scorer = RnnScore(num_q)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        s1 = self.scorer(x1, q)
        s2 = self.scorer(x2, q)
        return torch.sigmoid(s1 - s2)
