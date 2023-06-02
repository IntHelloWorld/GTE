import argparse
import os
import random
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))


def parse_args():
    description = "Add parameters"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data_dir", help="The path of raw data", default="/home/qyh/Desktop/github/dataset/CodeNet/Java_MINsample200_MAXline200_Accepted"
    )
    parser.add_argument("--output_dir", help="The path of output data", default="/home/qyh/Desktop/github/dataset/CodeNet")
    parser.add_argument("--ratio", help="The dataset split ratio, train:valid:test, require train+valid+test=10", default="8:1:1")
    args = parser.parse_args()
    return args


def move_submits(submits, dst_dir):
    if not dst_dir.exists():
        dst_dir.mkdir(parents=True)
    for submit in submits:
        os.system(f"cp {submit} {dst_dir}")


def make_dataset(data_dir, output_dir, ratio):
    print(f"Make dataset with ratio {ratio}")
    train, valid, test = map(lambda x: int(x), ratio.split(":"))
    input_dir = Path(data_dir).name
    output_dir = output_dir / (input_dir + f"_RATIO{train}-{valid}-{test}")
    assert train + valid + test == 10
    for p_dir in tqdm(list(data_dir.iterdir())):
        submits = list(p_dir.iterdir())
        random.shuffle(submits)
        one_part = int(len(submits) / 10)

        valid_submits = submits[: one_part * valid]
        move_submits(valid_submits, output_dir / "valid" / p_dir.name)

        test_submits = submits[one_part * valid : one_part * (test + valid)]
        move_submits(test_submits, output_dir / "test" / p_dir.name)

        train_submits = submits[one_part * (test + valid) :]
        move_submits(train_submits, output_dir / "train" / p_dir.name)

    print("Finished!")


if __name__ == "__main__":
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ratio = args.ratio
    make_dataset(data_dir, output_dir, ratio)
