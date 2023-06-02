import os
import random
from pathlib import Path

from tqdm import tqdm

# data_dir = Path("/home/qyh/Desktop/github/dataset/CodeNet/dataset_java_100_correct")
# parser = PyAstParser("/home/qyh/Desktop/github/GitHub/MTN-cue/tree-sitter/python-java-c-languages.so", "java")


def read_dataset_pt(dataset_dir, parser, token_vocab):
    def read_split(path):
        tree_set = []
        for p_dir in tqdm(path.iterdir()):
            assert p_dir.is_dir()
            label = int(p_dir.name)
            for submit in p_dir.iterdir():
                ast = parser.file2ast(str(submit))
                tree, pt = parser.tree_for_point(ast, token_vocab, p_batch=1)
                tree_set.append([tree, label, pt])
        return tree_set

    data_dir = Path(dataset_dir)
    testset = read_split(data_dir)

    print(f"test: {len(testset)}")
    return testset


def read_dataset(dataset_dir, parser):
    trainset = []
    validset = []
    testset = []
    train_dir = Path(dataset_dir) / "train"
    for p_dir in tqdm(train_dir.iterdir()):
        assert p_dir.is_dir()
        label = int(p_dir.name)
        for submit in p_dir.iterdir():
            ast = parser.file2ast(str(submit))
            trainset.append([ast.root_node, label])

    valid_dir = Path(dataset_dir) / "valid"
    for p_dir in tqdm(valid_dir.iterdir()):
        assert p_dir.is_dir()
        label = int(p_dir.name)
        for submit in p_dir.iterdir():
            ast = parser.file2ast(str(submit))
            validset.append([ast.root_node, label])

    test_dir = Path(dataset_dir) / "test"
    for p_dir in tqdm(test_dir.iterdir()):
        assert p_dir.is_dir()
        label = int(p_dir.name)
        for submit in p_dir.iterdir():
            ast = parser.file2ast(str(submit))
            testset.append([ast.root_node, label])

    random.shuffle(trainset)
    random.shuffle(validset)
    random.shuffle(testset)

    print(f"train: {len(trainset)}, valid: {len(validset)}, test: {len(testset)}")
    return trainset, validset, testset


def create_ccd_data(indexdir, treedict, vocablen, vocabdict, device):
    trainfile = open(indexdir + "/train.txt")
    validfile = open(indexdir + "/valid.txt")
    testfile = open(indexdir + "/test.txt")

    trainlist = trainfile.readlines()
    validlist = validfile.readlines()
    testlist = testfile.readlines()
    traindata = []
    validdata = []
    testdata = []
    print("train data")
    traindata = createpairdata(treedict, trainlist, device=device)
    print("valid data")
    validdata = createpairdata(treedict, validlist, device=device)
    print("test data")
    testdata = createpairdata(treedict, testlist, device=device)
    return traindata, validdata, testdata


def createpairdata(treedict, urllist, device):
    datalist = []
    countlines = 1
    for line in tqdm(urllist, desc="Create pair data"):
        # print(countlines)
        countlines += 1
        pairinfo = line.split()
        code1url = pairinfo[0]
        code2url = pairinfo[1]
        sign = pairinfo[2]
        if sign == "1":
            label = 1
        else:
            label = -1
        data1 = treedict[code1url]
        data2 = treedict[code2url]
        data = [[data1, data2], label]
        datalist.append(data)
    return datalist
