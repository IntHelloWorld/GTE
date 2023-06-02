from typing import List

import dgl
import torch
from dgl import DGLGraph
from dgl.heterograph import DGLBlock


class BlockSampler:
    def __init__(self):
        pass

    def sample(self, g: DGLGraph, n_layers: List[int], root_node: List[int] = (0,)) -> List[DGLBlock]:
        """Sampling a graph to several blocks. See <https://docs.dgl.ai/en/latest/guide_cn/minibatch-custom-sampler.html>.

        Args:
            g (dgl.DGLGraph): The DGLGraph.
            n_layers (int): Number of layers of the tree structure.
            root_node (int, optional): Index of the root node of the tree structure. Defaults to 0.

        Returns:
            List[DGLBlock]: The Blocks.
        """
        blocks = []
        seed_nodes = torch.tensor(root_node)
        for layer in range(max(n_layers)):
            frontier = dgl.in_subgraph(g, seed_nodes)
            block = dgl.to_block(frontier, seed_nodes)
            blocks.insert(0, block)
            new_seed_nodes = []
            for n in seed_nodes:
                predecessors = g.predecessors(int(n))
                if len(predecessors) > 0:
                    for predecessor in predecessors:
                        if predecessor != n:  # predecessors should not be itself
                            new_seed_nodes.append(predecessor)
            seed_nodes = torch.hstack(new_seed_nodes)
            # input_nodes = block.srcdata[dgl.NID]
            # print(input_nodes)
            # output_nodes = block.dstdata[dgl.NID]
            # print(output_nodes)

        return blocks
