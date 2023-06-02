import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(10000.0, torch.arange(0, d_model, 2).float() / d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        #  x: (seq_len, batch, d_model)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M ".format(round(model_size / 1e6)) + str(model_size)


class Transformer(nn.Module):
    def __init__(self, d_model, n_head, n_layers, dropout=0.5):
        super().__init__()
        d_ff = d_model * 2
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        print(f"Model size of Transformer {get_model_size(self)}.")

    def forward(self, src):
        #  x: (seq_len, batch, d_model)
        src = self.encoder(src)
        return src


class TransformerClf(nn.Module):
    def __init__(self, d_model, n_head, n_layers, dropout, vocab_size, n_label):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.0)
        self.transformer = Transformer(d_model, n_head, n_layers, dropout)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.classifier = nn.Linear(d_model, n_label)
        print(f"Model size of TransformerClf {get_model_size(self)}.")

    def forward(self, input_ids, labels=None, p_samples=None):
        # input_ids: (batch, seq_len)
        src = self.embedding(input_ids)
        src = src.transpose(0, 1)
        src = self.pos_encoder(src)
        src = self.transformer(src)
        src = src[0]
        if p_samples is not None:
            json_line = {
                "vec": src.data.squeeze().cpu().numpy().tolist(),
                "height": p_samples[0],
                "leaves": p_samples[1],
                "not_leaves": p_samples[2],
                "size": p_samples[3],
            }
        logits = self.classifier(src)
        prob = F.softmax(logits, dim=1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            if p_samples is not None:
                return loss, prob, json_line
            return loss, prob
        else:
            if p_samples is not None:
                return prob, json_line
            return prob


if __name__ == "__main__":
    model = TransformerClf(128, 4, 2, 0, 100, 10)
    input_ids = torch.randint(0, 100, (10, 5))
    print(model(input_ids))
