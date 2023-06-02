"""Torch modules for graph tree embedding network (GTE)."""
import copy
from typing import List
import numpy as np
import torch
import math
import dgl
from dgl import DGLGraph
from dgl import function as fn
from dgl.base import DGLError
from torch import nn
from models.ProbingSample import ProbingSample


class GTEProgramClassification(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, vocab_size: int, n_classes: int, n_layers: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.gte_conv = GTEConv(hidden_dim, num_heads, vocab_size, n_layers, dropout, bias)
        self.fc_classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, blocks, node_type):
        gte_out = self.gte_conv(blocks, node_type)
        out_feature = self.fc_classify(gte_out)
        return out_feature


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
        self.max_len = 1000
        print(f"Model size of {get_model_size(self)}.")

    def forward(self, src):
        #  x: (seq_len, batch, d_model)
        if src.shape[0] > self.max_len:
            src = src[: self.max_len]
        src = self.encoder(src)
        return src


class GTEConv(nn.Module):
    """TreeGATv2Conv based on `How Attentive are Graph Attention Networks?
    <https://arxiv.org/pdf/2105.14491.pdf>`
    """

    def __init__(self, hidden_dim: int, num_heads: int, vocab_size: int, n_layers: int, dropout: float, bias: bool):
        """Init function of class GTEConv.

        Args:
            hidden_dim (int): Model hidden feature size.
            num_heads (int): Number of heads in Multi-Head Attention.
            vocab_size (int): Vocabulary size.
        """
        super(GTEConv, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.transformer = Transformer(hidden_dim, num_heads, n_layers, dropout)
        print(f"Model size of {get_model_size(self)}.")

    def _reducer(self, nodes):
        """
        Reducer
        B : batch size
        L : length, == number of in edges
        D : hidden dim
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

            # horizontal encode
            src = msg.transpose(0, 1)  # (L, B, D)
            src = src[:-1]  # discard node type
            rst = self.transformer(src)  # output: (L, B, D)
            rst = torch.max(rst, dim=0).values

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
        rst_features = None
        out_samples = []
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
                block.update_all(message_func=fn.copy_u("node_feat", "e"), reduce_func=self._reducer)

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
