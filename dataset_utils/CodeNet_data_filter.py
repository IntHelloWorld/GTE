import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from utils.PyAstParser import PyAstParser


def parse_args():
    description = "Add parameters"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data_dir", help="The path of raw data", default="/home/qyh/Desktop/github/dataset/CodeNet/Project_CodeNet/data")
    parser.add_argument(
        "--meta_dir", help="The path of CodeNet meta data", default="/home/qyh/Desktop/github/dataset/CodeNet/Project_CodeNet/metadata"
    )
    parser.add_argument("--output_dir", help="The path of output data", default="/home/qyh/Desktop/github/dataset/CodeNet")
    parser.add_argument("--language", help="The program language of raw data", default="Java")
    parser.add_argument("--parser_mode", help="The program language mode of tree-sitter", default="java")
    parser.add_argument("--status", help="The expected status of submit, divided by comma", default="Accepted")
    parser.add_argument("--n_sample", help="The lower limit of the number of submits in a problem", default=200, type=int)
    parser.add_argument("--n_line", help="The higher limit of the number of lines in a sample", default=200, type=int)
    parser.add_argument("--n_workers", help="The number of thread workers", default=8, type=int)
    args = parser.parse_args()
    return args


def solve_problem(meta_f, data_dir, output_dir, obj_lg, parser_mode, status, n_sample, n_line):
    p = meta_f.stem  # problem index
    parser = PyAstParser("utils/python-java-c-languages.so", parser_mode)

    def submit_filter(line):
        split_l = line.split(",")

        # filter language
        lg = split_l[4]
        if lg != obj_lg:
            return False

        # filter status
        sts = split_l[7]
        if sts not in status:
            return False

        # filter parse error
        suffix = split_l[6]
        sub = split_l[0]
        fname = sub + "." + suffix
        sub_f = data_dir / p / lg / fname
        assert sub_f.exists()
        if parser.file2ast(str(sub_f)) is None:
            return False

        # filter line amount
        with open(sub_f) as f:
            if len(f.readlines()) > n_line:
                return False

        return True

    with open(meta_f) as f:
        obj_lines = list(filter(submit_filter, f.readlines()[1:]))
        if len(obj_lines) < n_sample:
            return
        else:
            for line in obj_lines:
                split_l = line.split(",")
                sub = split_l[0]
                lg = split_l[4]
                suffix = split_l[6]
                fname = sub + "." + suffix
                sub_f = data_dir / p / lg / fname
                obj_dir = output_dir / p
                if not obj_dir.exists():
                    obj_dir.mkdir(parents=True)
                cmd = f"cp {sub_f} {obj_dir}"
                if not (obj_dir / fname).exists():
                    os.system(cmd)


if __name__ == "__main__":
    args = parse_args()
    meta_dir = Path(args.meta_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    obj_lg = args.language
    parser_mode = args.parser_mode
    status = set(args.status.split(","))
    n_sample = args.n_sample
    n_line = args.n_line
    n_workers = args.n_workers
    info = f"{obj_lg}_MINsample{str(n_sample)}_MAXline{str(n_line)}_{'+'.join(status)}"
    dataset_dir = output_dir / info

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for meta_f in meta_dir.iterdir():
            pool.submit(solve_problem, meta_f, data_dir, dataset_dir, obj_lg, parser_mode, status, n_sample, n_line)

    # Rename problem dir and save map
    pool.shutdown(wait=True)
    p_list = [p for p in dataset_dir.iterdir()]
    p_dict = dict(zip(map(lambda x: x.name, p_list), map(lambda x: str(x), range(len(p_list)))))
    for p in p_list:
        new_p = p.parent / p_dict[p.name]
        os.system(f"mv {p} {new_p}")

    p_dict_dir = output_dir / f"problem-dict_{info}.npy"
    np.save(p_dict_dir, p_dict)
