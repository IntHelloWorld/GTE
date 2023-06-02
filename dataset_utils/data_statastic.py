from pathlib import Path
from pprint import pprint


def count_sample_number_per_problem(data_dir):
    res = {}
    for i in range(0, 500, 100):
        res[f"{i} - {i+100}"] = 0
    res["0"] = 0
    res[">500"] = 0

    total = 0
    for p in data_dir.iterdir():
        name = p.name
        count = len(list(p.iterdir()))
        total += count
        if count == 0:
            res["0"] += 1
            continue
        elif count > 500:
            res[">500"] += 1
            continue
        else:
            for i in range(0, 500, 100):
                if i < count <= i + 100:
                    res[f"{i} - {i+100}"] += 1
                    break
    print(f"sample amount: {total}")
    pprint(res)


def count_line_number_per_sample(data_dir):
    res = {}
    for i in range(0, 500, 100):
        res[f"{i} - {i+100}"] = 0
    res["0"] = 0
    res[">500"] = 0

    total = 0
    for p in data_dir.iterdir():
        name = p.name
        samples = list(p.iterdir())
        count = len(samples)
        total += count
        for sample in samples:
            with open(sample) as s:
                n_line = len(s.readlines())
                if n_line == 0:
                    res["0"] += 1
                    continue
                elif n_line > 500:
                    res[">500"] += 1
                    continue
                else:
                    for i in range(0, 500, 100):
                        if i < n_line <= i + 100:
                            res[f"{i} - {i+100}"] += 1
                            break
    print(f"sample amount: {total}")
    pprint(res)


if __name__ == "__main__":
    # data_dir = Path("/home/qyh/Desktop/github/dataset/CodeNet/Java_200_Accepted")
    # data_dir = Path("/home/qyh/Desktop/github/dataset/CodeNet/dataset_java_100_correct")
    data_dir = Path("/home/qyh/Desktop/github/dataset/CodeNet/Java_MINsample200_MAXline200_Accepted")

    count_sample_number_per_problem(data_dir)
    count_line_number_per_sample(data_dir)
