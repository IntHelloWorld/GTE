import sys
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parents[2]))
from utils.PyAstParser import PyAstParser


def create_vocab(data_dir, parser):
    asts = {}
    asts.update(parser.dir2asts(str(data_dir), tranverse=True))
    vocab = PyAstParser.asts2token_vocab(asts)
    return vocab, len(vocab)


def create_vocab_pt(data_dir, parser):
    asts = {}
    asts.update(parser.dir2asts(str(data_dir), tranverse=True))
    type_vocab = PyAstParser.asts2type_vocab(asts)
    return type_vocab


def parse_and_create_vocab(jsonl, parser, if_subtoken):
    asts = parser.jsonl2ast(jsonl)
    vocab = PyAstParser.asts2token_vocab(asts, subtoken=if_subtoken)
    asts = asts[str(Path(jsonl).parent)]
    return vocab, asts
