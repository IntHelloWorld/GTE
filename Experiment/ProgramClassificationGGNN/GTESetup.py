import os
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from dgl import batch as dgl_batch
from dgl.dataloading import GraphDataLoader
from pyfiglet import Figlet
from sacred import Experiment
from torch import optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[2]))

from dataset.PCDataset import PCDataset
from models.GTE_model_GGNN import GTEProgramClassification
from models.TreeSampler import BlockSampler
from utils.early_stopping import EarlyStopping
from utils.log import Logger
from utils.metrics import accuracy
from utils.timing import Timing

ex = Experiment(name="GTE", base_dir="../../..", interactive=False)


class GTESetup:
    def __init__(self):
        self.init_logger()
        self.init_data()
        self.init_model()
        self._init_optimizer()

    @ex.capture(prefix="training_setup")
    def init_logger(self, log_dir):
        self.run_id = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        log_path = Path(log_dir)
        if not log_path.exists():
            log_path.mkdir(parents=True)
        self.logger = Logger("ProgramClassification_GGNN", str(log_path / f"{self.run_id}_log.txt"))
        ascii = Figlet(font="isometric2")
        self.logger.info("\n" + ascii.renderText("GTE"))
        self.logger.info(f"Run ID {self.run_id}" + "\n")

    @ex.capture(prefix="data_setup")
    def init_data(self, dataset_dir, vocab_dir):
        self.logger.info("data setup...")
        self.logger.info("\n" + pformat(ex.current_run.config["data_setup"]))
        train_dir = os.path.join(dataset_dir, "train")
        valid_dir = os.path.join(dataset_dir, "valid")
        test_dir = os.path.join(dataset_dir, "test")
        self.train_dataset = PCDataset(train_dir)
        self.logger.info(f"train dataset loaded, size:{len(self.train_dataset)}")
        self.valid_dataset = PCDataset(valid_dir)
        self.logger.info(f"valid dataset loaded, size:{len(self.valid_dataset)}")
        self.test_dataset = PCDataset(test_dir)
        self.logger.info(f"test dataset loaded, size:{len(self.test_dataset)}")
        self.token_to_idx = pickle.load(open(vocab_dir, "rb"))
        self.vocab_size = len(self.token_to_idx)
        self.logger.info(f"vocabulary loaded, size:{self.vocab_size}")
        self.logger.info("Done!\n")

    @ex.capture(prefix="model_setup")
    def init_model(self, hidden_dim, n_classes, dropout, bias, checkpoint):
        self.logger.info("model setup...")
        self.logger.info("\n" + pformat(ex.current_run.config["model_setup"]))
        self.model = GTEProgramClassification(
            hidden_dim,
            self.vocab_size,
            n_classes,
            dropout,
            bias,
        )
        self.logger.info("model constructed")
        self.checkpoint = checkpoint
        if checkpoint:
            if Path(checkpoint).exists():
                self.model.load_state_dict(torch.load(checkpoint))
                self.logger.info(f"Loaded checkpoint model: {checkpoint}")
        self.logger.info("Done!\n")

    @ex.capture(prefix="optimizer_setup")
    def _init_optimizer(self, learning_rate, reg_scale, scheduler=None, scheduler_params=None, optimizer="Adam"):
        self.logger.info("optimizer setup...")
        self.logger.info("\n" + pformat(ex.current_run.config["optimizer_setup"]))
        if optimizer == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=reg_scale)
        elif optimizer == "AdamW":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=reg_scale)
        elif optimizer == "Momentum":
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=reg_scale, momentum=0.95, nesterov=True)
        self.logger.info(f"optimizer: {optimizer}")

        self.scheduler = None
        if scheduler == "OneCycleLR":
            self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, **scheduler_params)
        elif scheduler == "StepLR":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, **scheduler_params)
        elif scheduler == "MultiStepLR":
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, **scheduler_params)
        elif scheduler == "ExponentialLR":
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, **scheduler_params)

        self.logger.info(f"scheduler: {scheduler}")
        self.logger.info("Done!\n")

    @ex.capture(prefix="training_setup")
    def train(
        self,
        with_cuda,
        cuda_device_id,
        max_epoch,
        batch_size,
        random_seed,
        early_stopping_patience,
        loader_num_workers,
        pin_memory,
        model_dir,
    ):
        self.logger.info("training setup...")
        self.logger.info("\n" + pformat(ex.current_run.config["training_setup"]))

        # cuda
        if with_cuda:
            self.model = self.model.cuda(cuda_device_id)
            self.device = f"cuda:{cuda_device_id}"
        else:
            self.device = "cpu"

        # early stop
        early_stopping = EarlyStopping(self.run_id, early_stopping_patience, self.logger)

        # scheduler
        def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
            def lr_lambda(current_step: int):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

            return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0.1 * max_epoch, num_training_steps=max_epoch)

        # loss function
        loss_func = CrossEntropyLoss(reduction="sum")
        if with_cuda:
            loss_func = loss_func.to(self.device)

        # random seed
        torch.manual_seed(random_seed)
        random.seed(random_seed)

        # collate_fn function
        def collate(x):
            gs, labels, n_layers, node_types = zip(*x)
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
            blocks = [b for b in blocks]
            labels = torch.LongTensor(labels)
            return blocks, labels, batch_node_types

        # model validation
        def test_model(data_loader):
            self.model.eval()
            all_outs = []
            all_labels = []
            total = len(data_loader.dataset) // batch_size + 1
            with torch.no_grad():
                for blocks, labels, batch_node_types in tqdm(data_loader, total=total, desc="Testing model"):
                    if with_cuda:
                        blocks = [b.to(self.device) for b in blocks]
                        labels = labels.to(self.device)
                    # run model
                    rst_features = self.model(blocks, batch_node_types)
                    all_outs.append(rst_features)
                    all_labels.append(labels)

            all_outs = torch.cat(all_outs).to("cpu")
            targets = torch.hstack(all_labels).to("cpu")
            predict_labels = np.array(torch.max(all_outs, dim=1).indices)
            expect_labels = np.array(targets)
            return accuracy(predict_labels, expect_labels)

        self.logger.info("Starting Training\n")
        # training loop
        with Timing() as t_epoch:
            for epoch_id in range(max_epoch):
                self.logger.info(f"<Starting epoch {epoch_id}>")

                # Train
                self.logger.info("[Training]")
                self.logger.info("Loading train data...")
                train_loader = GraphDataLoader(
                    self.train_dataset,
                    collate_fn=collate,
                    batch_size=batch_size,
                    num_workers=loader_num_workers,
                    pin_memory=pin_memory,
                    drop_last=False,
                    shuffle=True,
                )
                total = len(self.train_dataset) // batch_size + 1
                self.logger.info("Done!")

                self.model.train()
                self.optimizer.zero_grad()
                inspect_count = 0
                batch_count = 0
                epoch_train_loss = 0
                predict_labels = []
                expect_labels = []
                with tqdm(total=total) as t_bar:
                    t_bar.set_description("Training")
                    for blocks, labels, batch_node_types in train_loader:
                        if with_cuda:
                            blocks = [b.to(self.device) for b in blocks]
                            labels = labels.to(self.device)

                        batch_count += 1
                        inspect_count += 1

                        # run model
                        rst_features = self.model(blocks, batch_node_types)
                        outs = rst_features.squeeze()

                        # calculate loss
                        loss = loss_func(outs, labels)
                        epoch_train_loss += loss.item()

                        # record train performance
                        predict_labels = np.concatenate((predict_labels, np.array(torch.max(outs.to("cpu"), dim=1).indices)))
                        expect_labels = np.concatenate((expect_labels, np.array(labels.to("cpu"))))
                        inspect_count = 0

                        t_bar.set_description(f"Training, loss:{loss:.2f}")
                        t_bar.update(1)

                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()
                train_acc = accuracy(predict_labels, expect_labels)
                self.logger.info(
                    f"training end, took {t_epoch.measure():.2f} seconds, train_acc:{train_acc:.4f}, current learning "
                    f"rate:{self.optimizer.param_groups[0]['lr']:.6f} "
                )

                # Validation
                self.logger.info("[validation]")
                valid_loader = GraphDataLoader(
                    self.valid_dataset,
                    collate_fn=collate,
                    batch_size=batch_size,
                    drop_last=False,
                    shuffle=True,
                    num_workers=loader_num_workers,
                    pin_memory=pin_memory,
                )
                acc = test_model(valid_loader)
                self.logger.info(f"accuracy:{acc:.4f}")
                self.logger.info(f"Done!")
                # early stop
                if not early_stopping.evaluate(acc):
                    self.logger.info(f"Early stop!")
                    break
                # checkpoint
                if early_stopping.checkpoint:
                    self.model_name = f"model_{self.run_id}_acc{early_stopping.best:.4f}.pt"
                    early_stopping.save_checkpoint(model_dir, self.model_name, self.model)

                self.logger.info(f"Epoch {epoch_id} end, avg batch loss: {epoch_train_loss / batch_count}, took {t_epoch.measure():.2f} seconds")
                self.logger.info("-" * 80)

        # test
        self.logger.info("*" * 50)
        self.logger.info(f"Start testing...")
        test_loader = GraphDataLoader(
            self.test_dataset,
            collate_fn=collate,
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
            num_workers=loader_num_workers,
            pin_memory=pin_memory,
        )
        with Timing() as t_test:
            self.model.load_state_dict(torch.load(str(Path(model_dir) / self.model_name)))
            # self.model.load_state_dict(torch.load(self.checkpoint))
            acc = test_model(test_loader)
            self.logger.info(f"accuracy:{acc:.4f}")
            self.logger.info(f"Test end, took {t_test.measure():.2f} seconds")
        self.logger.info("*" * 50)
