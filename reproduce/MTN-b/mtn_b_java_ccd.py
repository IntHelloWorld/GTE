# mtn-b for code clone detection
import os
import pickle
import random
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from create_vocab import parse_and_create_vocab
from log import Logger
from torch.autograd import Variable
from tqdm import tqdm

sys.path.append(str(Path(".").absolute()))
from dataset.read_dataset_java import create_ccd_data
from reproduce.utils.PyAstParser import PyAstParser

os.chdir(Path(__file__).parent)

dataset_dir = "/home/qyh/Desktop/github/dataset/BigCloneBench/data.jsonl"
index_dir = "/home/qyh/Desktop/github/dataset/BigCloneBench"
# data_dir = "/home/qyh/Desktop/github/GitHub/MTN-cue/test_dataset"
parser = PyAstParser("../../utils/python-java-c-languages.so", "java")

run_id = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

threshold = 0
num_epochs = 16
embedding_dim = 100
batch_size = 32
dropout_rate = 0.5
dropout = nn.Dropout(dropout_rate)

# parser = PyAstParser("tree-sitter/python-java-c-languages.so", "java")
# path = Path("/home/qyh/Desktop/github/dataset/CodeNet/dataset_Java_100/706/s275683867.java")
# tree = parser.file2ast(path)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logger = Logger("mtnb_ccd", file=f"mtn_ccd_{run_id}.log")

logger.info("creating vocabulary...")
vocabdict, asts = parse_and_create_vocab(dataset_dir, parser, False)
vocablen = len(vocabdict)
with open("token_to_index_ccd.pkl", "wb") as f:
    pickle.dump(vocabdict, f)
logger.info("finished")
CUDA = True


def Var(v):
    if CUDA:
        return Variable(v.cuda())
    else:
        return Variable(v)


empty = np.zeros((embedding_dim,), dtype="float32")
empty = Var(torch.LongTensor(empty))


class funcdeclblock(nn.Module):
    def __init__(self, dim):
        super(funcdeclblock, self).__init__()
        self.proj = nn.Linear(4 * dim, dim, bias=True)
        self.proj2 = nn.Linear(2 * dim, dim, bias=True)

    def forward(self, modifiers, return_type, identifier, formal_parameters, block):
        modifiers = modifiers.view(-1)
        return_type = return_type.view(-1)
        identifier = identifier.view(-1)
        formal_parameters = formal_parameters.view(-1)
        block = block.view(-1)
        out1 = torch.cat([modifiers, return_type, identifier, formal_parameters], 0)  # Concatentate along depth
        out1 = F.relu(self.proj(out1))
        out2 = torch.cat([out1, block], 0)
        out = F.relu(self.proj2(out2))
        return out


class WhileBlock(nn.Module):
    def __init__(self, dim):
        super(WhileBlock, self).__init__()
        self.proj = nn.Linear(2 * dim, dim, bias=True)

    def forward(self, x, y):
        x = x.view(-1)
        y = y.view(-1)
        out = torch.cat([x, y], 0)  # Concatentate along depth
        out = F.relu(self.proj(out))
        return out


class DoWhileBlock(nn.Module):
    def __init__(self, dim):
        super(DoWhileBlock, self).__init__()
        self.proj = nn.Linear(2 * dim, dim, bias=True)

    def forward(self, x, y):
        x = x.view(-1)
        y = y.view(-1)
        out = torch.cat([x, y], 0)  # Concatentate along depth
        out = F.relu(self.proj(out))
        return out


class ForBlock(nn.Module):
    def __init__(self, dim):
        super(ForBlock, self).__init__()
        self.proj1 = nn.Linear(3 * dim, dim, bias=True)
        self.proj2 = nn.Linear(2 * dim, dim, bias=True)

    def forward(self, init, cond, next, block):
        init = init.view(-1)
        cond = cond.view(-1)
        next = next.view(-1)
        block = block.view(-1)
        control = torch.cat([init, cond, next], 0)
        control = F.relu(self.proj1(control))
        y = torch.cat([control, block], 0)
        y = F.relu(self.proj2(y))
        return y


class EnhancedForBlock(nn.Module):
    def __init__(self, dim):
        super(EnhancedForBlock, self).__init__()
        self.proj1 = nn.Linear(3 * dim, dim, bias=True)
        self.proj2 = nn.Linear(2 * dim, dim, bias=True)

    def forward(self, type, item, iter, block):
        type = type.view(-1)
        item = item.view(-1)
        iter = iter.view(-1)
        block = block.view(-1)
        control = torch.cat([type, item, iter], 0)
        control = F.relu(self.proj1(control))
        y = torch.cat([control, block], 0)
        y = F.relu(self.proj2(y))
        return y


class IfBlock(nn.Module):
    def __init__(self, dim):
        super(IfBlock, self).__init__()
        self.proj = nn.Linear(2 * dim, dim, bias=True)
        self.proj2 = nn.Linear(2 * dim, dim, bias=True)

    def forward(self, control, then_stmt, else_stmt=None):
        control = control.view(-1)
        then_stmt = then_stmt.view(-1)
        out1 = torch.cat([control, then_stmt], 0)
        out1 = F.relu(self.proj(out1))
        if else_stmt is None:  # no else or else if statement
            return out1
        else_stmt = else_stmt.view(-1)
        out2 = torch.cat([control, else_stmt], 0)
        out2 = F.relu(self.proj(out2))
        out = torch.max(out1, out2)
        return out


class SwitchBlock(nn.Module):
    def __init__(self, dim):
        super(SwitchBlock, self).__init__()
        self.proj = nn.Linear(2 * dim, dim, bias=True)

    def forward(self, x, y):
        x = x.view(-1)
        y = y.view(-1)
        out = torch.cat([x, y], 0)  # Concatentate along depth
        out = F.relu(self.proj(out))
        return out


class CaseBlock(nn.Module):
    def __init__(self, dim):
        super(CaseBlock, self).__init__()
        self.hidden_dim = dim
        self.lstm = nn.LSTM(dim, dim)
        self.hidden = self.init_hidden()
        self.proj = nn.Linear(2 * dim, dim, bias=True)

    def init_hidden(self):
        return (
            Var(torch.zeros(1, 1, self.hidden_dim)),
            Var(torch.zeros(1, 1, self.hidden_dim)),
        )

    def forward(self, switch_label, stmts):
        if len(stmts) > 0:
            stmts = torch.cat(stmts)
            stmts = F.torch.unsqueeze(stmts, 1)
            stmts_out, self.hidden = self.lstm(stmts, (self.hidden[0].detach(), self.hidden[1].detach()))
            stmts_out = stmts_out[-1].view(-1)
        else:
            stmts_out = Var(torch.zeros(1, self.hidden_dim)).view(-1)
        switch_label = switch_label.view(-1)
        out = torch.cat([switch_label, stmts_out], 0)
        out = F.relu(self.proj(out))
        return out


class SeqBlock(nn.Module):
    def __init__(self, dim):
        super(SeqBlock, self).__init__()
        self.hidden_dim = dim
        self.lstm = nn.LSTM(dim, dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (
            Var(torch.zeros(1, 1, self.hidden_dim)),
            Var(torch.zeros(1, 1, self.hidden_dim)),
        )

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x, (self.hidden[0].detach(), self.hidden[1].detach()))
        out = lstm_out[-1]
        out = out.view(-1)
        return out


class treelstm(nn.Module):
    def __init__(self, dim, mem_dim):
        super(treelstm, self).__init__()
        self.ix = nn.Linear(dim, mem_dim)
        self.ih = nn.Linear(mem_dim, mem_dim)
        self.fh = nn.Linear(mem_dim, mem_dim)
        self.fx = nn.Linear(dim, mem_dim)
        self.ux = nn.Linear(dim, mem_dim)
        self.uh = nn.Linear(mem_dim, mem_dim)
        self.ox = nn.Linear(dim, mem_dim)
        self.oh = nn.Linear(mem_dim, mem_dim)

    def forward(self, inputs, child_c, child_h, child_h_sum):
        i = torch.sigmoid(self.ix(inputs) + self.ih(child_h_sum))
        o = torch.sigmoid(self.ox(inputs) + self.oh(child_h_sum))
        u = torch.tanh(self.ux(inputs) + self.uh(child_h_sum))
        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = F.torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = torch.sigmoid(f)
        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)
        c = F.torch.mul(i, u) + F.torch.sum(fc, 0)
        h = F.torch.mul(o, torch.tanh(c))
        return c, h


class MTN_b(nn.Module):  # the MTN classifier
    def __init__(self, dim, mem_dim):
        super(MTN_b, self).__init__()
        self.dim = dim
        self.mem_dim = mem_dim

        self.fdeclb = funcdeclblock(dim=dim)
        self.whileb = WhileBlock(dim=dim)
        self.dowhileb = DoWhileBlock(dim=dim)
        self.switchb = SwitchBlock(dim=dim)
        self.caseb = CaseBlock(dim=dim)
        self.forb = ForBlock(dim=dim)
        self.enforb = EnhancedForBlock(dim=dim)
        self.ifb = IfBlock(dim=dim)
        self.seqb = SeqBlock(dim=dim)
        self.fdecll = treelstm(dim, mem_dim)
        self.whilel = treelstm(dim, mem_dim)
        self.dowhilel = treelstm(dim, mem_dim)
        self.forl = treelstm(dim, mem_dim)
        self.forcl = treelstm(dim, mem_dim)
        self.enforcl = treelstm(dim, mem_dim)
        self.ifl = treelstm(dim, mem_dim)
        self.switchl = treelstm(dim, mem_dim)
        self.casel = treelstm(dim, mem_dim)
        self.seql = treelstm(dim, mem_dim)
        self.otherl = treelstm(dim, mem_dim)
        # self.fc2 = nn.Linear(dim, n_class)
        self.embeddings = nn.Embedding(vocablen, embedding_dim)

    def node_forward(self, inputs, child_c, child_h, node):
        empty = Var(torch.zeros(1, self.mem_dim))

        def get_specified_h(node, child_h, type, vague=False):
            _, id = PyAstParser.get_child_with_type(node, type, vague=vague)
            if id is not None:
                return child_h[id]
            else:
                return empty

        nodetype = node.type
        childisseq = False
        if nodetype == "block":
            childisseq = True

        # MTN modular networks for computing child_sum.
        if type(child_h) == list:
            is_else = False
            if nodetype == "method_declaration":
                modifiers = get_specified_h(node, child_h, "modifiers")
                return_type = get_specified_h(node, child_h, r"_type", vague=True)
                identifier = get_specified_h(node, child_h, "identifier")
                formal_parameters = get_specified_h(node, child_h, "formal_parameters")
                block = get_specified_h(node, child_h, "block")
                child_h_sum = self.fdeclb(modifiers, return_type, identifier, formal_parameters, block)
            elif nodetype == "while_statement":
                child_h_sum = self.whileb(child_h[1], child_h[2])
            elif nodetype == "do_statement":
                child_h_sum = self.dowhileb(child_h[1], child_h[3])
            elif nodetype == "switch_expression":
                child_h_sum = self.switchb(child_h[1], child_h[2])
            elif nodetype == "switch_block_statement_group":
                child_h_sum = self.caseb(child_h[0], child_h[2:])
            elif nodetype == "for_statement":
                for_type = PyAstParser.distinguish_for(node)
                if for_type == "":
                    child_h_sum = self.forb(empty, empty, empty, child_h[-1])
                elif for_type == "i":
                    child_h_sum = self.forb(child_h[2], empty, empty, child_h[-1])
                elif for_type == "c":
                    child_h_sum = self.forb(empty, child_h[3], empty, child_h[-1])
                elif for_type == "u":
                    child_h_sum = self.forb(empty, empty, child_h[4], child_h[-1])
                elif for_type == "ic":
                    child_h_sum = self.forb(child_h[2], child_h[3], empty, child_h[-1])
                elif for_type == "iu":
                    child_h_sum = self.forb(child_h[2], empty, child_h[4], child_h[-1])
                elif for_type == "cu":
                    child_h_sum = self.forb(empty, child_h[3], child_h[5], child_h[-1])
                else:  # for_type == "icu"
                    child_h_sum = self.forb(child_h[2], child_h[3], child_h[5], child_h[-1])
            elif nodetype == "enhanced_for_statement":
                child_h_sum = self.enforb(child_h[2], child_h[3], child_h[5], child_h[7])
            elif nodetype == "if_statement":
                if_type = PyAstParser.distinguish_if(node)
                if if_type == "if":  # no else or else if statement
                    child_h_sum = self.ifb(child_h[1], child_h[2], else_stmt=None)
                else:  #  has else or else if statement
                    child_h_sum = self.ifb(child_h[1], child_h[2], child_h[4])
            else:
                is_else = True
                child_h = torch.cat(child_h)
                child_h = F.torch.unsqueeze(child_h, 1)
                if childisseq == True:  # statements of 'statements' or 'body'.
                    child_h_sum = self.seqb(child_h)
                else:
                    child_h_sum = F.torch.sum(torch.squeeze(child_h, 1), 0)
            if not is_else:
                child_h = torch.cat(child_h)
                child_h = F.torch.unsqueeze(child_h, 1)
        else:  # leaf node
            child_h_sum = F.torch.sum(torch.squeeze(child_h, 1), 0)

        child_c = F.torch.unsqueeze(child_c, 1)

        # Child sum TreeLSTMs for computing cell-state and hidden_state.
        if nodetype == "method_declaration":
            c, h = self.fdecll(inputs, child_c, child_h, child_h_sum)
        elif nodetype == "while_statement":
            c, h = self.whilel(inputs, child_c, child_h, child_h_sum)
        elif nodetype == "do_statement":
            c, h = self.dowhilel(inputs, child_c, child_h, child_h_sum)
        elif nodetype == "switch_expression":
            c, h = self.switchl(inputs, child_c, child_h, child_h_sum)
        elif nodetype == "switch_block_statement_group":
            c, h = self.casel(inputs, child_c, child_h, child_h_sum)
        elif nodetype == "for_statement":
            c, h = self.forl(inputs, child_c, child_h, child_h_sum)
        elif nodetype == "enhanced_for_statement":
            c, h = self.enforcl(inputs, child_c, child_h, child_h_sum)
        elif nodetype == "if_statement":
            c, h = self.ifl(inputs, child_c, child_h, child_h_sum)
        else:
            if childisseq == True:
                c, h = self.seql(inputs, child_c, child_h, child_h_sum)
            else:
                c, h = self.otherl(inputs, child_c, child_h, child_h_sum)
        return c, h

    def traverse(self, node):
        word = PyAstParser.get_token(node, lower=False)  # use identifier names
        currentinput = self.embeddings(Var(torch.LongTensor([vocabdict[word]])))
        if node.named_child_count == 0:  # leaf node
            child_c = Var(torch.zeros(1, self.mem_dim))
            child_h = Var(torch.zeros(1, self.mem_dim))
        else:
            children = node.children
            childcs = []
            childhs = []
            for child in children:
                if "comment" in child.type:
                    continue
                if not child.is_named:  # initialize unnamed child embeddings
                    empty_c = Var(torch.zeros(1, self.mem_dim))
                    empty_h = Var(torch.zeros(1, self.mem_dim))
                    childcs.append(empty_c)
                    childhs.append(empty_h)
                    continue
                c, h = self.traverse(child)
                childcs.append(c)
                childhs.append(h)
            child_c = torch.cat(childcs)
            child_h = childhs

        currentc, currenth = self.node_forward(currentinput, child_c, child_h, node)
        return currentc, currenth

    def forward(self, x):
        out = self.traverse(x.root_node)[1]
        out = out.view(1, -1)
        # out = self.fc2(out)
        return out


def create_batches(data):
    # random.shuffle(data)
    batches = [data[graph : graph + batch_size] for graph in range(0, len(data), batch_size)]
    return batches


if __name__ == "__main__":
    model = MTN_b(embedding_dim, embedding_dim)
    model.load_state_dict(torch.load("/home/qyh/projects/GTE/reproduce/MTN-b/output/mtnb_ccd2023-03-05--20-05-58 - epoch1.pt"))
    criterion = nn.MSELoss()
    if CUDA == True:
        model = model.cuda()
        criterion = criterion.cuda()
    device = torch.device("cuda:0")

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    logger.info("reading dataset...")
    trainset, validset, testset = create_ccd_data(index_dir, asts, vocablen, vocabdict, device)

    if not os.path.exists("output"):
        os.mkdir("output")
    if not os.path.exists("results"):
        os.mkdir("results")

    def test(dataset):
        model.eval()
        with torch.no_grad():
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            results = []
            for data, label in tqdm(dataset, desc="validation"):
                label = torch.tensor(label, dtype=torch.float, device=device)
                prediction1 = model(data[0])
                prediction2 = model(data[1])
                output = F.cosine_similarity(prediction1, prediction2)
                results.append(output.item())
                prediction = torch.sign(output).item()

                if prediction > threshold and label.item() == 1:
                    tp += 1
                if prediction <= threshold and label.item() == -1:
                    tn += 1
                if prediction > threshold and label.item() == -1:
                    fp += 1
                if prediction <= threshold and label.item() == 1:
                    fn += 1
            logger.info(f"TP:{tp}, TN:{tn}, FP:{fp}, FN:{fn}")
            if tp + fp == 0:
                logger.info("precision is none")
                return results, 0
            p = tp / (tp + fp)
            if tp + fn == 0:
                logger.info("recall is none")
                return results, 0
            r = tp / (tp + fn)
            f1 = 2 * p * r / (p + r)
            logger.info(f"precision:{p}, recall:{r}, F1:{f1}")
            return results, f1

    best_f1 = 0
    main_index = 0
    logger.info("mtn-b-java")
    for epoch in range(num_epochs):
        model.train()
        logger.info("epoch: " + str(epoch + 1))
        random.shuffle(trainset)
        totalloss = 0.0
        logger.info(f"train data: {len(trainset)}")
        batches = create_batches(trainset)
        for batch in tqdm(batches, total=len(batches), desc=f"Training"):
            optimizer.zero_grad()
            batchloss = 0
            for data, label in batch:
                label = torch.tensor(label, dtype=torch.float, device=device)
                prediction1 = model(data[0])
                prediction2 = model(data[1])
                cossim = F.cosine_similarity(prediction1, prediction2)
                batchloss = batchloss + criterion(cossim, label)
            batchloss.backward(retain_graph=True)
            optimizer.step()
            loss = batchloss.item()
            totalloss += loss
            main_index = main_index + len(batch)
        avg_loss = totalloss / main_index
        logger.info(f"Epoch {epoch} (Avg Loss={avg_loss})")
        torch.save(model.state_dict(), f"output/mtnb_ccd{run_id}_epoch{epoch}.pt")

    # validation
    devresults, f1 = test(validset)
    if f1 > best_f1:
        torch.save(model.state_dict(), f"output/mtnb_ccd{run_id}.pt")
        logger.info(f"best f1: ({best_f1})->({f1})")
        best_f1 = f1

    # test
    testresults, _ = test(testset)
    resfile = open(f"results/mtnb_ccd_{run_id}_test_result.txt", mode="w")
    for res in testresults:
        resfile.write(str(res) + "\n")
    resfile.close()
