"""Preprocess raw data for probing Network task"""
import argparse
import pickle
import sys
from pathlib import Path

import torch
from dgl.data.utils import save_graphs, save_info
from tqdm import tqdm

proj_path = Path(__file__).absolute().parents[1]
sys.path.append(str(proj_path))
from utils.PyAstParser import PyAstParser


def parse_args():
    description = "Add parameters"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dataset_dir",
        help="The path of the dataset",
        default="/home/qyh/dataset/Project_CodeNet_Java250_RATIO6-2-2_ProbingTask",
    )
    parser.add_argument("--language", help="The program language of parser", default="java")
    parser.add_argument(
        "--token_vocab", help="The path of input token vocab", default="/home/qyh/projects/GTE/vocabulary/token_to_index_PC_java250.pkl"
    )
    parser.add_argument("--output_dir", help="The path of output data", default="/home/qyh/dataset")
    args = parser.parse_args()
    return args


def save_graphs_and_infos(graphs, output_dir):
    """
    Save graphs (include DGLGraph and node token ids) and graph informations (include node_type and n_layers).
    """
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    info = {}
    gs = []
    labels = []
    n_graph = 0
    for dir_name in tqdm(graphs, desc="save graphs and infos..."):
        label = int(Path(dir_name).stem)
        for f_name, g in graphs[dir_name].items():
            f_path = dir_name + "," + f_name
            gs.append(g["graph"])
            if len(info) == 0:
                info["n_layers"] = [g["n_layers"]]
                info["node_types"] = [g["node_types"]]
                info["p_samples"] = [g["p_samples"]]
                info["f_paths"] = [f_path]
            else:
                info["n_layers"].append(g["n_layers"])
                info["node_types"].append(g["node_types"])
                info["p_samples"].append(g["p_samples"])
                info["f_paths"].append(f_path)
        n = len(graphs[dir_name])
        labels.extend([label] * n)
        n_graph += n

    labels = {"labels": torch.ShortTensor(labels)}

    # save graphs
    graph_path = output_dir / f"DGLGraph_Num{n_graph}.bin"
    save_graphs(str(graph_path), gs, labels)

    # save infos
    info_path = output_dir / f"Info_Num{n_graph}.pkl"
    save_info(str(info_path), info)


if __name__ == "__main__":
    args = parse_args()
    parser = PyAstParser(str(proj_path / "utils/python-java-c-cpp-languages.so"), args.language)

    print("build and save vocabulary...")
    token_vocab = pickle.load(open(args.token_vocab, "rb"))

    print("generate and save graphs...")
    data_dir = Path(args.dataset_dir)
    for split_dir in data_dir.iterdir():
        split_name = split_dir.name

        print(f"generate {split_name} set...")
        split_asts = parser.dir2asts(str(split_dir), tranverse=True)
        graphs = parser.trees2DGLgraphs_for_probing(split_asts, token_vocab, p_batch=1)

        print(f"save {split_name} set...")
        output_dir = Path(args.output_dir)
        assert output_dir.exists()
        suffix = "_DGL"
        output_dir = output_dir / (data_dir.name + suffix) / split_name
        save_graphs_and_infos(graphs, output_dir)
