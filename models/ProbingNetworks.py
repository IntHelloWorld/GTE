from torch import nn


class ProbingRegressionnNetwork(nn.Module):
    def __init__(self, input_dim: int, dense_dim: int, bias: bool = True):
        super().__init__()
        self.dense = nn.Linear(input_dim, dense_dim, bias=bias)
        self.fc_classify = nn.Linear(dense_dim, 1, bias=bias)

    def forward(self, vec):
        out = self.dense(vec)
        out = self.fc_classify(out)
        return out


class ProbingClassificationNetwork(nn.Module):
    def __init__(self, input_dim: int, dense_dim: int, n_classes: int, bias: bool = True):
        super().__init__()
        self.dense = nn.Linear(input_dim, dense_dim, bias=bias)
        self.fc_classify = nn.Linear(dense_dim, n_classes, bias=bias)

    def forward(self, vec):
        out = self.dense(vec)
        out = self.fc_classify(out)
        return out
