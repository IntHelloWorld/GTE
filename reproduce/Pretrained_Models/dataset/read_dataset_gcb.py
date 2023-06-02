from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from tree_sitter import Language, Parser

from dataset.gcb_parser import (
    DFG_java,
    DFG_python,
    index_to_code_token,
    remove_comments_and_docstrings,
    tree_to_token_index,
)

dfg_function = {
    "python": DFG_python,
    "java": DFG_java,
}

# load parsers
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language("dataset/gcb_parser/my-languages.so", lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


# remove comments, tokenize code and extract dataflow
def extract_dataflow(code, parser, lang):
    # remove comments
    try:
        code = remove_comments_and_docstrings(code, lang)
    except:
        pass
    # obtain dataflow
    if lang == "php":
        code = "<?php" + code + "?>"
    try:
        tree = parser[0].parse(bytes(code, "utf8"))
        root_node = tree.root_node
        tokens_index = tree_to_token_index(root_node)
        code = code.split("\n")
        code_tokens = [index_to_code_token(x, code) for x in tokens_index]
        index_to_code = {}
        for idx, (index, code) in enumerate(zip(tokens_index, code_tokens)):
            index_to_code[index] = (idx, code)
        try:
            DFG, _ = parser[1](root_node, index_to_code, {})
        except:
            DFG = []
        DFG = sorted(DFG, key=lambda x: x[1])
        indexs = set()
        for d in DFG:
            if len(d[-1]) != 0:
                indexs.add(d[1])
            for x in d[-1]:
                indexs.add(x)
        new_DFG = []
        for d in DFG:
            if d[1] in indexs:
                new_DFG.append(d)
        dfg = new_DFG
    except:
        dfg = []
    return code_tokens, dfg


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(
        self,
        input_tokens,
        input_ids,
        position_idx,
        dfg_to_code,
        dfg_to_dfg,
    ):
        # The first code function
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.position_idx = position_idx
        self.dfg_to_code = dfg_to_code
        self.dfg_to_dfg = dfg_to_dfg


def convert_code_to_features(file, tokenizer, args):
    # source
    with file.open(encoding="utf-8") as f:
        code = f.read()

    # extract data flow
    parser = parsers[args.lang]
    code_tokens, dfg = extract_dataflow(code, parser, args.lang)
    code_tokens = [tokenizer.tokenize("@ " + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]

    # truncating
    code_tokens = code_tokens[: args.code_length + args.data_flow_length - 3 - min(len(dfg), args.data_flow_length)][: 512 - 3]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
    dfg = dfg[: args.code_length + args.data_flow_length - len(source_tokens)]
    source_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    source_ids += [tokenizer.unk_token_id for x in dfg]
    padding_length = args.code_length + args.data_flow_length - len(source_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    source_ids += [tokenizer.pad_token_id] * padding_length

    # reindex
    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]
    return InputFeatures(source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg)


class GCBdataset(Dataset):
    def __init__(self, args, dataset_dir, tokenizer):
        super(GCBdataset, self).__init__()
        self.args = args
        self.examples = []
        dataset_dir = Path(dataset_dir)

        for p_dir in tqdm(dataset_dir.iterdir()):
            assert p_dir.is_dir()
            label = int(p_dir.name)
            for submit in p_dir.iterdir():
                input_tensor = convert_code_to_features(submit, tokenizer, args)
                self.examples.append([input_tensor, label])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        in_feats = self.examples[item][0]
        # calculate graph-guided masked function
        attn_mask_1 = np.zeros((self.args.code_length + self.args.data_flow_length, self.args.code_length + self.args.data_flow_length), dtype=bool)
        # calculate begin index of node and max length of input
        node_index = sum([i > 1 for i in in_feats.position_idx])
        max_length = sum([i != 1 for i in in_feats.position_idx])
        # sequence can attend to sequence
        attn_mask_1[:node_index, :node_index] = True
        # special tokens attend to all tokens
        for idx, i in enumerate(in_feats.input_ids):
            if i in [0, 2]:
                attn_mask_1[idx, :max_length] = True
        # nodes attend to code tokens that are identified from
        for idx, (a, b) in enumerate(in_feats.dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask_1[idx + node_index, a:b] = True
                attn_mask_1[a:b, idx + node_index] = True
        # nodes attend to adjacent nodes
        for idx, nodes in enumerate(in_feats.dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(in_feats.position_idx):
                    attn_mask_1[idx + node_index, a + node_index] = True

        label = self.examples[item][1]
        return (
            torch.tensor(in_feats.input_ids),
            torch.tensor(in_feats.position_idx),
            torch.tensor(attn_mask_1),
            torch.tensor(label),
        )
