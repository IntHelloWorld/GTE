from torch import nn

from models.GTE_model_GRU import GTEConv as GRUGTEConv
from models.GTE_model_Transformer import GTEConv as TransformerGTEConv
from models.GTE_model_TreeLSTM import GTEConv as TreeLSTMGTEConv
from models.GTE_model_Transformer_relative_pos import GTEConv as TransformerRelativePosGTEConv
from models.GTE_model_Transformer_no_type import GTEConv as TransformerNoTypeGTEConv
from models.GTE_model_GGNN import GTEConv as GGNNConv


class GTEProbingTaskCollection(nn.Module):
    def __init__(
        self,
        model_type: str,
        hidden_dim: int,
        num_heads: int,
        vocab_size: int,
        n_classes: int,
        n_layers: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        if model_type == "Transformer":
            self.gte_conv = TransformerGTEConv(hidden_dim, num_heads, vocab_size, n_layers, dropout, bias)
        elif model_type == "Transformer_relative_pos":
            self.gte_conv = TransformerRelativePosGTEConv(hidden_dim, num_heads, vocab_size, n_layers, dropout, bias)
        elif model_type == "Transformer_no_type":
            self.gte_conv = TransformerNoTypeGTEConv(hidden_dim, num_heads, vocab_size, n_layers, dropout, bias)
        elif model_type == "TreeLSTM":
            self.gte_conv = TreeLSTMGTEConv(hidden_dim, vocab_size)
        elif model_type == "GRU":
            self.gte_conv = GRUGTEConv(hidden_dim, vocab_size, dropout, bias, n_layers)
        elif model_type == "GGNN":
            self.gte_conv = GGNNConv(hidden_dim, vocab_size, dropout, bias)

        self.fc_classify = nn.Linear(hidden_dim, n_classes, bias=bias)

    def forward(self, blocks, node_type, p_samples=None, get_attn=False):
        gte_out, samples = self.gte_conv(blocks, node_type, p_samples)
        out = self.fc_classify(gte_out)
        return out, samples
