"""Torch modules for graph tree embedding network (GTE)."""
from typing import List

import torch
import dgl
from dgl import DGLGraph
from dgl import function as fn
from dgl.base import DGLError
from torch import nn
from models.ProbingSample import ProbingSample


class GTEProgramClassification(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        n_classes: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        self.gte_conv = GTEConv(hidden_dim, vocab_size, dropout, bias)
        self.fc_classify = nn.Linear(hidden_dim, n_classes, bias=bias)

    def forward(self, blocks, node_type):
        gte_out = self.gte_conv(blocks, node_type)
        out_feature = self.fc_classify(gte_out)
        return out_feature


class GTEConv(nn.Module):
    """TreeGATv2Conv based on `How Attentive are Graph Attention Networks?
    <https://arxiv.org/pdf/2105.14491.pdf>`
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        dropout: float = 0.0,
        bias: bool = True,
    ):
        """Init function of class TreeGATv2Conv

        Args:
            hidden_dim (int): Model hidden feature size.
            num_heads (int): Number of heads in Multi-Head Attention.
            n_classes (int): Number of problem classes.
            vocab_size (int): Vocabulary size.
            feat_drop (float, optional): Dropout rate on feature. Defaults to 0.0.
            attn_drop (float, optional): Dropout rate on attention weight.. Defaults to 0.0.
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
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.GRU = nn.GRU(hidden_dim, hidden_dim, batch_first=False, bias=bias)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _gru_reducer(self, nodes):
        """
        LSTM reducer
        B : batch size
        L : length, == number of in edges
        D : hidden dim
        N : 2 if bidirectional else 1
        """
        msg = nodes.mailbox["e"]  # (B, L, D)
        length = msg.shape[1]
        if length == 1:
            # messages come from the nodes with self loop, need not calculation
            return {"ft": msg.squeeze(1)}
        else:
            # The order of message features decided by the order of edges, while the order of edges decided by the
            # edge-adding order when constructing the graph, as we adopt function 'add_self_loop' after building the
            # graph, the message feature comes from the self-loop edge is always the last one.
            h = msg[:, :-1, :]  # (B, L-1, D)
            h = h.contiguous()

            current_input = msg[:, -1, :]  # (B, D)
            current_input = current_input.contiguous().unsqueeze(0)  # (1, B, D)

            f_sum = torch.sum(h, dim=1).unsqueeze(0)  # (1, B, D)

            # horizontal GRU
            output, h_out = self.GRU(current_input, f_sum)

            # layer norm
            rst = self.layer_norm(h_out[-1])
            rst = self.dropout(rst)

            # rst = h_out[-1]
            return {"ft": rst}

    def forward(self, blocks: List[DGLGraph], node_type: List[str], p_samples=None):
        """Compute graph attention network layer.

        Args:
            blocks (List[DGLGraph]): The blocks. See `https://docs.dgl.ai/en/latest/guide_cn/minibatch-custom-sampler.html`.
            node_type (List[str]): A list ecording the type of each node, the elements are type strings (e.g., "if", "while"...).

        Raises:
            DGLError: If there are 0-in-degree nodes in the input graph, it will raise DGLError since no message will be passed to those nodes. This will cause invalid output.

        Returns:
            result (torch.Tensor): The output feature of shape :math:`(N, H, D_{out})` where :math:`H` is the number of heads, and :math:`D_{out}` is size of output feature.
            attention (torch.Tensor) (optional): The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of edges. This is returned only when :attr:`get_attention` is ``True``.
        """
        rst_features = None
        out_samples = []
        for idx, block in enumerate(blocks):
            if rst_features is None:
                src_feat = block.srcdata["token_id"]
            else:
                src_feat = rst_features
            dst_feat = block.dstdata["token_id"]

            if idx == 0:
                first_layer = True
            else:
                first_layer = False

            with block.local_scope():
                if (block.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, output for those nodes will be invalid. "
                        "This is harmful for some applications, causing silent performance regression. "
                        "Adding self-loop on the input graph by calling `g = dgl.add_self_loop(g)` will resolve the issue. "
                    )
                # embedding layer and fc layer for destination nodes features
                h_dst = self.embeddings(dst_feat)
                h_dst = h_dst.squeeze(1)

                # embedding layer and fc layer for source nodes features
                if first_layer:
                    node_feat = self.embeddings(src_feat)
                    node_feat = node_feat.squeeze(1)
                else:
                    node_feat = torch.cat((h_dst, src_feat), dim=0)

                block.srcdata.update({"el": node_feat})  # (num_src_node, num_heads, hidden_dim)

                # message passing & reducing
                block.update_all(message_func=fn.copy_u("el", "e"), reduce_func=self._gru_reducer)
                rst = block.dstdata["ft"]

                rst_features = rst

                # collect probing samples
                if p_samples is not None:
                    dst_ids = block.dstdata[dgl.NID].cpu().numpy().tolist()
                    for node_id in dst_ids:
                        if node_id in p_samples:
                            info = p_samples[node_id]
                            sample = ProbingSample(
                                rst_features[dst_ids.index(node_id)].data.cpu().numpy().tolist(),
                                info[0],
                                info[1],
                                info[2],
                                info[3],
                            )
                            out_samples.append(sample)

        if p_samples is not None:
            return rst_features, out_samples

        return rst_features  # (batch_size, hidden_dim)
