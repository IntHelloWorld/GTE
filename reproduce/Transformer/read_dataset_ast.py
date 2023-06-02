from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parents[2]))
from utils.PyAstParser import PyAstParser


def get_sequence(node, sequence):
    token = PyAstParser.get_token(node)
    children = PyAstParser.get_children(node)
    if token:
        sequence.append(token)
    for child in children:
        get_sequence(child, sequence)


def get_leaves_sequence(node, sequence):
    token = PyAstParser.get_token(node)
    children = PyAstParser.get_children(node)
    if token and len(children) == 0:
        sequence.append(token)
    for child in children:
        get_leaves_sequence(child, sequence)


def convert_codes_to_features(code, vocab, args):
    # source
    code_tokens = [vocab[c] for c in code]
    source_ids = code_tokens[: args.block_size]
    padding_length = args.block_size - len(source_ids)
    source_ids += [0] * padding_length
    return torch.tensor(source_ids)


class CodeDatasetProbing(Dataset):
    def __init__(self, args, dataset_dir, vocab):
        super(CodeDatasetProbing, self).__init__()
        self.data = []
        dataset_dir = Path(dataset_dir)
        parser = PyAstParser(args.so_file, args.lang)

        for p_dir in tqdm(dataset_dir.iterdir()):
            assert p_dir.is_dir()
            label = int(p_dir.name)
            for submit in p_dir.iterdir():
                ast = parser.file2ast(str(submit))
                code_seq = []
                get_sequence(ast.root_node, code_seq)
                tree_pt, p_samples = parser.tree_for_point(ast, p_batch=1)
                input_tensor = convert_codes_to_features(code_seq, vocab, args)
                self.data.append([input_tensor, label, p_samples])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class CodeDataset(Dataset):
    def __init__(self, args, dataset_dir, vocab):
        super(CodeDataset, self).__init__()
        self.data = []
        dataset_dir = Path(dataset_dir)
        parser = PyAstParser(args.so_file, args.lang)

        for p_dir in tqdm(dataset_dir.iterdir()):
            assert p_dir.is_dir()
            label = int(p_dir.name)
            for submit in p_dir.iterdir():
                ast = parser.file2ast(str(submit))
                code_seq = []
                get_sequence(ast.root_node, code_seq)
                input_tensor = convert_codes_to_features(code_seq, vocab, args)
                self.data.append([input_tensor, label])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class CodeDatasetLeaves(Dataset):
    def __init__(self, args, dataset_dir, vocab):
        super(CodeDatasetLeaves, self).__init__()
        self.data = []
        dataset_dir = Path(dataset_dir)
        parser = PyAstParser(args.so_file, args.lang)

        for p_dir in tqdm(dataset_dir.iterdir()):
            assert p_dir.is_dir()
            label = int(p_dir.name)
            for submit in p_dir.iterdir():
                ast = parser.file2ast(str(submit))
                code_seq = []
                get_leaves_sequence(ast.root_node, code_seq)
                input_tensor = convert_codes_to_features(code_seq, vocab, args)
                self.data.append([input_tensor, label])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
