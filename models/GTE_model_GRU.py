"""Torch modules for graph tree embedding network (GTE)."""
from typing import List

import dgl
import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl import function as fn
from dgl.base import DGLError
from models.ProbingSample import ProbingSample
from torch import nn


class GTEProgramClassification(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        n_classes: int,
        dropout: float = 0.0,
        bias: bool = True,
        n_layers: int = 1,
    ):
        super().__init__()
        self.gte_conv = GTEConv(hidden_dim, vocab_size, dropout, bias, n_layers)
        self.fc_classify = nn.Linear(hidden_dim, n_classes, bias=bias)

    def forward(self, blocks, node_type):
        gte_out = self.gte_conv(blocks, node_type)
        # take the c_h as the final output
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
        n_layers: int = 1,
    ):
        """Init function of GTEConv

        Args:
            hidden_dim (int): Model hidden feature size.
            vocab_size (int): Vocabulary size.
            dropout (float, optional): Dropout rate on feature. Defaults to 0.0.
            bias (bool, optional): If set to False, the layer will not learn an additive bias. Defaults to True.
            n_layers (int, optional): Layers amount of the LSTM. Defaults to 1.
        """
        super(GTEConv, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.layer_norm = nn.LayerNorm(normalized_shape=hidden_dim)
        self.horizon_GRU = nn.GRU(hidden_dim, hidden_dim, n_layers, batch_first=True, bias=bias)
        self.dropout = nn.Dropout(dropout)
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
            output_horizon, h_horizon = self.horizon_GRU(h, current_input)
            rst = h_horizon.squeeze(0)

            # layer norm
            rst = self.dropout(rst)
            rst = self.layer_norm(rst)
            return {"ft": rst}

    def forward(self, blocks: List[DGLGraph], node_type: List[str], p_samples=None):
        """Compute graph attention network layer.

        Args:
            blocks (List[DGLGraph]): The blocks. See `https://docs.dgl.ai/en/latest/guide_cn/minibatch-custom-sampler.html`.
            node_type (List[str]): A list contains the type of each node, the elements are type strings (e.g., "if", "while"...).

        Raises:
            DGLError: If there are 0-in-degree nodes in the input graph, it will raise DGLError since no message will be passed to those nodes. This will cause invalid output.

        Returns:
            result (torch.Tensor): The output feature of shape :math:`(N, H, D_{out})` where :math:`H` is the number of heads, and :math:`D_{out}` is size of output feature.
            attention (torch.Tensor) (optional): The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of edges. This is returned only when :attr:`get_attention` is ``True``.
        """
        out_samples = []
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
                    node_feat = feat_src.squeeze(1)
                else:
                    # update source node feature
                    feat_dst = self.embeddings(block.dstdata["token_id"])
                    feat_dst = feat_dst.squeeze(1)
                    node_feat = torch.cat((feat_dst, rst_features), dim=0)

                block.srcdata.update({"node_feat": node_feat})  # (num_nodes, hidden_dim*2)

                # message passing & reducing
                block.update_all(message_func=fn.copy_u("node_feat", "e"), reduce_func=self._gru_reducer)

                rst = block.dstdata["ft"]  # (num_tgt_node, hidden_dim*2)
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
        return rst_features
