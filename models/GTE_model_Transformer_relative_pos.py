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


class RelativePosition(nn.Module):
    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()  # [len_q, len_k, num_units]

        return embeddings


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.max_relative_position = 4
        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = None

    def forward(self, query, key, value, mask=None):
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
        batch_size = query.shape[0]
        len_k = key.shape[1]
        len_q = query.shape[1]
        len_v = value.shape[1]

        query = self.fc_q(query)
        key = self.fc_k(key)
        value = self.fc_v(value)

        r_q1 = query.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        r_k1 = key.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        attn1 = torch.matmul(r_q1, r_k1.permute(0, 1, 3, 2))

        r_q2 = query.permute(1, 0, 2).contiguous().view(len_q, batch_size * self.n_heads, self.head_dim)
        r_k2 = self.relative_position_k(len_q, len_k)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(batch_size, self.n_heads, len_q, len_k)
        if self.scale is None:
            self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(attn1.device)
        attn = (attn1 + attn2) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)

        attn = self.dropout(torch.softmax(attn, dim=-1))

        # attn = [batch size, n heads, query len, key len]
        r_v1 = value.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        weight1 = torch.matmul(attn, r_v1)
        r_v2 = self.relative_position_v(len_q, len_v)
        weight2 = attn.permute(2, 0, 1, 3).contiguous().view(len_q, batch_size * self.n_heads, len_k)
        weight2 = torch.matmul(weight2, r_v2)
        weight2 = weight2.transpose(0, 1).contiguous().view(batch_size, self.n_heads, len_q, self.head_dim)

        x = weight1 + weight2
        # x = [batch size, n heads, query len, head dim]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, query len, n heads, head dim]
        x = x.view(batch_size, -1, self.hid_dim)
        # x = [batch size, query len, hid dim]
        x = self.fc_o(x)
        # x = [batch size, query len, hid dim]
        return x


class TransformerEncoderLayer(nn.Module):
    __constants__ = ["batch_first"]

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttentionLayer(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class Transformer(nn.Module):
    def __init__(self, d_model, n_head, n_layers, dropout=0.5):
        super().__init__()
        d_ff = d_model * 2
        self.max_len = 1000
        encoder_layers = TransformerEncoderLayer(d_model, n_head, d_ff, dropout)
        self.encoder = TransformerEncoder(encoder_layers, n_layers)
        print(f"Model size of {get_model_size(self)}.")

    def forward(self, src):
        #  src: (batch, seq_len d_model)
        if src.shape[1] > self.max_len:
            print(f"Input length {src.shape[1]} exceeds maximum length, truncated to {self.max_len}.")
            src = src[:, : self.max_len, :].contiguous()
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
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-5)
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
            rst = self.transformer(msg)
            # rst = torch.sum(rst, dim=0)  # (B, D): sum up all the message features
            # rst = self.norm(rst)
            # rst = torch.mean(rst, dim=0)
            rst = torch.max(rst, dim=1).values
            # rst = rst[-1]  # (B, D): return the vec correspond to the parent node

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
