from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def convert_codes_to_features(file, tokenizer, args):
    # source
    with file.open(encoding="utf-8") as f:
        code = f.read()
    code_tokens = tokenizer.tokenize(code)[: args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return torch.tensor(source_ids)


class CodeDataset(Dataset):
    def __init__(self, args, dataset_dir, tokenizer):
        super(CodeDataset, self).__init__()
        self.data = []
        dataset_dir = Path(dataset_dir)

        for p_dir in tqdm(dataset_dir.iterdir()):
            assert p_dir.is_dir()
            label = int(p_dir.name)
            for submit in p_dir.iterdir():
                input_tensor = convert_codes_to_features(submit, tokenizer, args)
                self.data.append([input_tensor, label])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def read_dataset_pt(dataset_dir, parser, token_vocab, type_vocab):
    def read_split(path):
        tree_set = []
        for p_dir in tqdm(path.iterdir()):
            assert p_dir.is_dir()
            label = int(p_dir.name)
            for submit in p_dir.iterdir():
                ast = parser.file2ast(str(submit))
                tree, pt = parser.tree_for_point(ast, token_vocab, type_vocab, p_batch=1)
                tree_set.append([tree, label, pt])
        return tree_set

    data_dir = Path(dataset_dir)
    testset = read_split(data_dir)

    print(f"test: {len(testset)}")
    return testset
