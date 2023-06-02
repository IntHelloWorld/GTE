import torch
import torch.nn as nn

from torch.nn import CrossEntropyLoss


class GRUModel(nn.Module):
    def __init__(self, d_model, tokenizer, n_label):
        super().__init__()
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.embedding = nn.Embedding(tokenizer.vocab_size, d_model)
        self.classifier = nn.Linear(d_model, n_label)

    def forward(self, input_ids=None, labels=None):
        src = self.embedding(input_ids)
        _, h_n = self.gru(src)
        h_n = h_n.squeeze(0)
        logits = self.classifier(h_n)
        prob = torch.softmax(logits, -1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob
