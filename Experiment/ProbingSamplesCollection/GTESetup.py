import json
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from pprint import pformat
from random import shuffle

import dgl
import numpy as np
import torch
from dgl import batch as dgl_batch
from dgl.dataloading import GraphDataLoader
from pyfiglet import Figlet
from sacred import Experiment
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[2]))

from dataset.PTDataset import PTDataset
from models.ProbingTaskCollection import GTEProbingTaskCollection
from models.TreeSampler import BlockSampler
from utils.log import Logger
from utils.metrics import accuracy

ex = Experiment(name="GTE", base_dir="../../..", interactive=False)


class GTESetup:
    def __init__(self):
        self.init_logger()
        self.init_data()
        self.init_model()

    @ex.capture(prefix="training_setup")
    def init_logger(self, log_dir, model_type):
        self.run_id = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        self.model_type = model_type
        log_path = Path(log_dir)
        if not log_path.exists():
            log_path.mkdir(parents=True)
        self.logger = Logger(f"ProbingTaskCollection_{model_type}", str(log_path / f"{self.run_id}_log.txt"))
        ascii = Figlet(font="isometric2")
        self.logger.info("\n" + ascii.renderText("GTE"))
        self.logger.info(f"Run ID {self.run_id}" + "\n")

    @ex.capture(prefix="data_setup")
    def init_data(self, dataset_dir, output_dir, token_vocab_dir):
        self.logger.info("data setup...")
        self.logger.info("\n" + pformat(ex.current_run.config["data_setup"]))
        self.dataset = PTDataset(dataset_dir)
        self.logger.info(f"dataset loaded, size:{len(self.dataset)}")
        self.token_to_idx = pickle.load(open(token_vocab_dir, "rb"))
        self.vocab_size = len(self.token_to_idx)
        self.logger.info(f"token vocabulary loaded, size:{self.vocab_size}")
        self.output_dir = output_dir
        if not Path(output_dir).exists():
            Path(output_dir).mkdir(parents=True)
        self.logger.info("Done!\n")

    @ex.capture(prefix="model_setup")
    def init_model(self, hidden_dim, num_heads, n_classes, dropout, bias, n_layers, checkpoint):
        self.logger.info("model setup...")
        self.logger.info("\n" + pformat(ex.current_run.config["model_setup"]))
        self.model = GTEProbingTaskCollection(self.model_type, hidden_dim, num_heads, self.vocab_size, n_classes, n_layers, dropout, bias)
        self.logger.info(f"{self.model_type} model constructed")
        self.model.load_state_dict(torch.load(checkpoint))
        self.logger.info(f"Loaded checkpoint model: {checkpoint}")
        self.logger.info("Done!\n")

    @ex.capture(prefix="training_setup")
    def train(self, with_cuda, cuda_device_id, batch_size, random_seed, loader_num_workers, pin_memory):
        self.logger.info("generating setup...")
        self.logger.info("\n" + pformat(ex.current_run.config["training_setup"]))

        # cuda
        if with_cuda:
            self.model = self.model.cuda(cuda_device_id)
            self.device = f"cuda:{cuda_device_id}"
        else:
            self.device = "cpu"

        # random seed
        torch.manual_seed(random_seed)
        random.seed(random_seed)

        # collate_fn function
        def collate(x):
            gs, labels, n_layers, node_types, p_samples = zip(*x)
            # batched graphs
            bg = dgl_batch(gs)
            batch_node_types = []
            for node_list in node_types:
                batch_node_types.extend(node_list)
            # calculate root node ids
            root_nodes = [0]
            for i in range(len(gs) - 1):
                root_nodes.append(gs[i].num_nodes() + root_nodes[i])
            assert len(root_nodes) == len(gs)
            # transfer graph to blocks
            sampler = BlockSampler()
            blocks = sampler.sample(bg, n_layers, root_nodes)
            blocks[-1].dstdata[dgl.NID]
            blocks = [b for b in blocks]
            labels = torch.LongTensor(labels)
            return blocks, labels, batch_node_types, p_samples[0]

        # def collate(x):
        #     g, label, n_layer, node_types, p_samples = x[0]
        #     # transfer graph to blocks
        #     sampler = BlockSampler()
        #     blocks = sampler.sample(g, [n_layer], [0])
        #     blocks = [b for b in blocks]
        #     label = torch.LongTensor([label])
        #     return blocks, label, node_types, p_samples

        self.logger.info("[generating probing task samples]")
        data_loader = GraphDataLoader(
            self.dataset,
            collate_fn=collate,
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=loader_num_workers,
            pin_memory=pin_memory,
        )

        self.model.eval()
        all_outs = []
        all_labels = []
        total = len(data_loader.dataset) // batch_size + 1
        output_file = str(Path(self.output_dir) / f"probing_task_{self.model_type}.jsonl")
        with torch.no_grad():
            for blocks, label, node_types, p_samples in tqdm(data_loader, total=total, desc="Testing model"):
                if with_cuda:
                    blocks = [b.to(self.device) for b in blocks]
                    label = label.to(self.device)
                # run model and collect probing task samples
                out, out_samples = self.model(blocks, node_types, p_samples)
                if len(out_samples) > 1:
                    shuffle(out_samples)
                with open(output_file, "a") as f:
                    for sample in out_samples:
                        json.dump(sample.get_json(), f)
                        f.write("\n")

                all_outs.append(out)
                all_labels.append(label.item())

        all_outs = torch.stack(all_outs).squeeze().to("cpu")
        predict_labels = np.array(torch.max(all_outs, dim=1).indices)
        expect_labels = np.array(all_labels)
        acc = accuracy(predict_labels, expect_labels)
        self.logger.info(f"model accuracy: {acc:.4f}")
