"""Torch modules for graph tree embedding network (GTE)."""
from typing import List

import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl import function as fn
from dgl.base import DGLError
from torch import nn


class GTEProgramClassification(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        vocab_size: int,
        subtoken: bool,
        n_classes: int,
        feedforward_dim: 2048,
        feat_drop: float = 0.0,
        attn_drop: float = 0.0,
        negative_slope: float = 0.2,
        residual: bool = True,
        bias: bool = True,
        share_weights: bool = False,
        modular_weights: bool = True,
        reducer_type: str = "lstm",
        lstm_n_layers: int = 2,
    ):
        super().__init__()
        self.gte_conv = GTEConv(
            hidden_dim,
            num_heads,
            vocab_size,
            subtoken,
            feedforward_dim,
            feat_drop,
            attn_drop,
            negative_slope,
            residual,
            bias,
            share_weights,
            modular_weights,
            reducer_type,
            lstm_n_layers,
        )
        self.fc_classify = nn.Linear(hidden_dim, n_classes, bias=bias)

    def forward(self, blocks, node_type, get_attn=False):
        gte_out, attn = self.gte_conv(blocks, node_type, get_attn)
        # take the c_h as the final output
        out_feature = self.fc_classify(gte_out)
        return out_feature, attn


class GTEConv(nn.Module):
    """TreeGATv2Conv based on `How Attentive are Graph Attention Networks?
    <https://arxiv.org/pdf/2105.14491.pdf>`
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        vocab_size: int,
        subtoken: bool,
        feedforward_dim: 2048,
        feat_drop: float = 0.0,
        attn_drop: float = 0.0,
        negative_slope: float = 0.2,
        residual: bool = False,
        bias: bool = True,
        share_weights: bool = False,
        modular_weights: bool = False,
        reducer_type: str = "sum",
        lstm_n_layers: int = 1,
    ):
        """Init function of class TreeGATv2Conv

        Args:
            hidden_dim (int): Model hidden feature size.
            num_heads (int): Number of heads in Multi-Head Attention.
            n_classes (int): Number of problem classes.
            vocab_size (int): Vocabulary size.
            feat_drop (float, optional): Dropout rate on feature. Defaults to 0.0.
            attn_drop (float, optional): Dropout rate on attention weight. Defaults to 0.0.
            negative_slope (float, optional): LeakyReLU angle of negative slope. Defaults to 0.2.
            residual (bool, optional): If True, use residual connection. Defaults to False.
            activation (optional):callable activation function/layer or None, optional.
                    If not None, applies an activation function to the updated node features. Defaults to None.
            bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
            share_weights (bool, optional): If set to True, the same matrix for :math:`W_{left}` and :math:`W_{right}` in
                    the above equations, will be applied to the source and the target node of every edge. Defaults to False.
            modular_weights (bool, optional): If set to True, modular matrix will be applied to the node before
                    computing message function. Defaults to False.
            reducer_type (str, optional): Type of the reduce function, choose between "sum" and "lstm". Defaults to "sum".
            lstm_n_layers (int, optional): Layers amount of the LSTM. Defaults to 1.
        """
        super(GTEConv, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.subtoken = subtoken
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)
        self.bidirectional = False
        self.lstm_n_layers = lstm_n_layers
        self.horizon_GRU = nn.GRU(hidden_dim, hidden_dim, lstm_n_layers, batch_first=True, bidirectional=self.bidirectional, bias=bias)
        self.hidden_combine_layer = nn.Linear(hidden_dim * 2, hidden_dim, bias=bias)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.activation = nn.ReLU()
        self.negative_slope = negative_slope
        self.share_weights = share_weights
        self.modular_weights = modular_weights
        self.bias = bias
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        nn.init.kaiming_normal_(self.embeddings.weight)
        self.horizon_GRU.reset_parameters()

    def _gru_reducer(self, nodes):
        """
        LSTM reducer
        B : batch size
        L : length, == number of in edges
        D : hidden dim
        N : 2 if bidirectional else 1
        """
        msg = nodes.mailbox["e"]  # (B, L, D*2)
        length = msg.shape[1]
        if length == 1:
            # messages come from the nodes with self loop, need not calculation
            return {"ft": msg.squeeze(1)}
        else:
            # The order of message features decided by the order of edges, while the order of edges decided by the
            # edge-adding order when constructing the graph, as we adopt function 'add_self_loop' after building the
            # graph, the message feature comes from the self-loop edge is always the last one.
            h = msg[:, :-1, :]  # (B, L-1, D*2)
            h = h.contiguous()

            current_input = msg[:, -1, :]  # (B, D*2)
            current_input = current_input.contiguous().unsqueeze(0)  # (N, B, D)

            # horizontal encode
            output_horizon, h_horizon = self.horizon_GRU(h)
            h_horizon = h_horizon.squeeze(0)

            # layer norm
            rst = self.layer_norm(h_horizon)
            # rst_h = h_out[-1]
            return {"ft": rst}

    def forward(self, blocks: List[DGLGraph], node_type: List[str], get_attention=False):
        """Compute graph attention network layer.

        Args:
            blocks (List[DGLGraph]): The blocks. See `https://docs.dgl.ai/en/latest/guide_cn/minibatch-custom-sampler.html`.
            node_type (List[str]): A list contains the type of each node, the elements are type strings (e.g., "if", "while"...).
            get_attention (bool, optional): If output the attention. Defaults to False.

        Raises:
            DGLError: If there are 0-in-degree nodes in the input graph, it will raise DGLError since no message will be passed to those nodes. This will cause invalid output.

        Returns:
            result (torch.Tensor): The output feature of shape :math:`(N, H, D_{out})` where :math:`H` is the number of heads, and :math:`D_{out}` is size of output feature.
            attention (torch.Tensor) (optional): The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of edges. This is returned only when :attr:`get_attention` is ``True``.
        """
        attentions = []
        rst_features = None
        for idx, block in enumerate(blocks):
            if idx == 0:
                first_layer = True
            else:
                first_layer = False

            with block.local_scope():
                if (block.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, output for those nodes will be invalid. "
                        "This is harmful for some applications, causing silent performance regression. "
                        "Adding self-loop on the input graph by calling `g = dgl.add_self_loop(g)` will resolve the "
                        "issue. "
                    )
                # embedding layer for nodes features
                if first_layer:
                    feat_src = self.embeddings(block.srcdata["token_id"])
                    if self.subtoken:
                        feat_src = torch.mean(feat_src, dim=1)
                    else:
                        feat_src = feat_src.squeeze(1)
                    node_feat = self.feat_drop(feat_src)
                else:
                    # update source node feature
                    feat_dst = self.embeddings(block.dstdata["token_id"])
                    if self.subtoken:
                        feat_dst = torch.mean(feat_dst, dim=1)
                    else:
                        feat_dst = feat_dst.squeeze(1)
                    node_feat = torch.cat((feat_dst, rst_features), dim=0)
                    node_feat = self.feat_drop(node_feat)

                block.srcdata.update({"node_feat": node_feat})  # (num_nodes, hidden_dim*2)

                # message passing & reducing
                block.update_all(message_func=fn.copy_u("node_feat", "e"), reduce_func=self._gru_reducer)

                rst = block.dstdata["ft"]  # (num_tgt_node, hidden_dim*2)
                rst_features = rst

        return rst_features, None
