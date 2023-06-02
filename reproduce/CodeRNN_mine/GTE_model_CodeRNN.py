"""Torch modules for graph tree embedding network (GTE)."""
from typing import List

import torch
import torch.nn.functional as F
from dgl import DGLGraph
from dgl import function as fn
from dgl.base import DGLError
from torch import nn


class GTEProgramClassification(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, n_classes: int):
        super().__init__()
        self.gte_conv = GTEConv(hidden_dim, vocab_size)
        self.fc_classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, blocks, node_type):
        gte_out = self.gte_conv(blocks, node_type)
        out_feature = self.fc_classify(gte_out)
        return out_feature


class CodeRNNCell(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, inputs, child_h):
        child_h_sum = torch.sum(child_h, 0)
        x = inputs + F.relu(self.W(child_h_sum))
        return x


class GTEConv(nn.Module):
    """TreeGATv2Conv based on `How Attentive are Graph Attention Networks?
    <https://arxiv.org/pdf/2105.14491.pdf>`
    """

    def __init__(self, hidden_dim: int, vocab_size: int):
        super(GTEConv, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.codeRNN = CodeRNNCell(hidden_dim)

    def _codeRNN_reducer(self, nodes):
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
            msg_t = msg.transpose(0, 1)
            h = msg_t[:-1, :, :]  # (L-1, B, D)
            h = h.contiguous()

            current_input = msg_t[-1, :, :]  # (B, D)
            current_input = current_input.contiguous()  # (B, D)

            # horizontal encode
            h_out = self.codeRNN(current_input, h)

            return {"ft": h_out}

    def forward(self, blocks: List[DGLGraph], node_type: List[str]):
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
                block.update_all(message_func=fn.copy_u("node_feat", "e"), reduce_func=self._codeRNN_reducer)

                rst = block.dstdata["ft"]  # (num_tgt_node, hidden_dim*2)
                rst_features = rst

        return rst_features
