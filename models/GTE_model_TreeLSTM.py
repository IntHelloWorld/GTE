"""Torch modules for graph tree embedding network (GTE)."""
from typing import List

import torch
import dgl
import torch.nn.functional as F
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
    ):
        super().__init__()
        self.gte_conv = GTEConv(hidden_dim, vocab_size)
        self.fc_classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, blocks, node_type):
        gte_out = self.gte_conv(blocks, node_type)
        # take the c_h as the final output
        out_feature = self.fc_classify(gte_out)
        return out_feature


class treelstm(nn.Module):
    def __init__(self, dim, mem_dim):
        super(treelstm, self).__init__()
        self.ix = nn.Linear(dim, mem_dim)
        self.ih = nn.Linear(mem_dim, mem_dim)
        self.fh = nn.Linear(mem_dim, mem_dim)
        self.fx = nn.Linear(dim, mem_dim)
        self.ux = nn.Linear(dim, mem_dim)
        self.uh = nn.Linear(mem_dim, mem_dim)
        self.ox = nn.Linear(dim, mem_dim)
        self.oh = nn.Linear(mem_dim, mem_dim)

    def forward(self, inputs, child_c, child_h, child_h_sum):
        i = torch.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = torch.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = torch.tanh(self.ux(inputs) + self.uh(child_h_sum))
        fx = self.fx(inputs)
        f = F.torch.cat([torch.unsqueeze(self.fh(child_hi) + fx, 0) for child_hi in child_h], 0)
        f = torch.sigmoid(f)
        fc = F.torch.squeeze(F.torch.mul(f, child_c))
        c = F.torch.mul(i, u) + F.torch.sum(fc, 0)
        h = F.torch.mul(o, torch.tanh(c))
        return c, h


class GTEConv(nn.Module):
    """TreeGATv2Conv based on `How Attentive are Graph Attention Networks?
    <https://arxiv.org/pdf/2105.14491.pdf>`
    """

    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
    ):
        super(GTEConv, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=hidden_dim)
        self.tree_LSTM = treelstm(hidden_dim, hidden_dim)

    def _lstm_reducer(self, nodes):
        """
        LSTM reducer
        B : batch size
        L : length, == number of in edges
        D : hidden dim
        N : 2 if bidirectional else 1
        """
        h_c = nodes.mailbox["e"]  # (B, L, D*2)
        length = h_c.shape[1]
        if length == 1:
            # messages come from the nodes with self loop, need not calculation
            return {"ft": h_c.squeeze(1)}
        else:
            # The order of message features decided by the order of edges, while the order of edges decided by the
            # edge-adding order when constructing the graph, as we adopt function 'add_self_loop' after building the
            # graph, the message feature comes from the self-loop edge is always the last one.
            batch_size = h_c.shape[0]
            h_c = h_c.transpose(0, 1)
            h_c_no_loop = h_c[:-1, :, :]  # (L-1, B, D*2)
            h, c = torch.split(h_c_no_loop, self.hidden_dim, dim=2)  # (L-1, B, D)
            h = h.contiguous()
            c = c.contiguous()
            child_h_sum = F.torch.sum(h, 0)  # (B, D)

            current_input = h_c[-1, :, :]  # (B, D*2)
            current_input, _ = torch.split(current_input, self.hidden_dim, dim=1)
            current_input = current_input.contiguous()  # (B, D)

            # tree LSTM
            c_out, h_out = self.tree_LSTM(current_input, c, h, child_h_sum)
            c_out = self.layer_norm_1(c_out)
            h_out = self.layer_norm_2(h_out)

            rst = torch.cat([h_out, c_out], dim=1)  # (D*2)
            return {"ft": rst}

    def forward(self, blocks: List[DGLGraph], node_type: List[str], p_samples=None):
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
                    feat_src = feat_src.squeeze(1)
                    node_feat = F.pad(feat_src, (0, self.hidden_dim))
                else:
                    # update source node feature
                    feat_dst = self.embeddings(block.dstdata["token_id"])
                    feat_dst = feat_dst.squeeze(1)
                    feat_dst = F.pad(feat_dst, (0, self.hidden_dim))
                    node_feat = torch.cat((feat_dst, rst_features), dim=0)

                block.srcdata.update({"node_feat": node_feat})  # (num_nodes, hidden_dim*2)

                # message passing & reducing
                block.update_all(message_func=fn.copy_u("node_feat", "e"), reduce_func=self._lstm_reducer)

                rst = block.dstdata["ft"]  # (num_tgt_node, hidden_dim*2)
                rst_features = rst

                # collect probing samples
                if p_samples is not None:
                    dst_ids = block.dstdata[dgl.NID].cpu().numpy().tolist()
                    for node_id in dst_ids:
                        if node_id in p_samples:
                            info = p_samples[node_id]
                            vec = torch.split(rst_features[dst_ids.index(node_id)], self.hidden_dim, dim=0)[0]
                            sample = ProbingSample(
                                vec.data.cpu().numpy().tolist(),
                                info[0],
                                info[1],
                                info[2],
                                info[3],
                            )
                            out_samples.append(sample)

        if idx == len(blocks) - 1:
            # if last block, split and output the feature
            h_out, c_out = torch.split(rst_features, self.hidden_dim, dim=1)
            h_out = h_out.contiguous()
            c_out = c_out.contiguous()
            if p_samples is not None:
                return h_out, out_samples
            return h_out
        return rst_features
