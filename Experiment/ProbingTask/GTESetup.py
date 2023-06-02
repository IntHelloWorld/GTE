import json
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from pyfiglet import Figlet
from sacred import Experiment
from torch import optim
from torch.nn import L1Loss, CrossEntropyLoss
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[2]))

from models.ProbingNetworks import ProbingRegressionnNetwork, ProbingClassificationNetwork
from utils.early_stopping import EarlyStopping
from utils.log import Logger
from utils.metrics import accuracy
from utils.timing import Timing

ex = Experiment(name="GTECodeCloneDetection", base_dir="../../..", interactive=False)


class GTESetup:
    def __init__(self):
        self.init_logger()
        self.init_data()
        self.init_model()

    @ex.capture(prefix="training_setup")
    def init_logger(self, log_dir, batch_size):
        self.batch_size = batch_size
        self.run_id = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        log_path = Path(log_dir)
        if not log_path.exists():
            log_path.mkdir(parents=True)
        self.logger = Logger("experiment", str(log_path / f"{self.run_id}_log.txt"))
        ascii = Figlet(font="isometric2")
        self.logger.info("\n" + ascii.renderText("GTE"))
        self.logger.info(f"Run ID {self.run_id}" + "\n")

    @ex.capture(prefix="data_setup")
    def init_data(self, data_dir):
        self.logger.info("data setup...")
        self.logger.info("\n" + pformat(ex.current_run.config["data_setup"]))
        samples = []
        with open(data_dir) as f:
            vec, high, leav, not_leav, size = [], [], [], [], []
            for line in tqdm(f, desc="reading jsonl data"):
                line = line.strip()
                js = json.loads(line)
                vec.append(torch.FloatTensor(np.asarray(js["vec"])))
                high.append(int(js["height"]))
                leav.append(int(js["leaves"]))
                not_leav.append(int(js["not_leaves"]))
                size.append(int(js["size"]))
                if len(high) == self.batch_size:
                    sample = {
                        "vec": torch.stack(vec),
                        "height": torch.FloatTensor(high),
                        "leaves": torch.LongTensor(leav),
                        "not_leaves": torch.LongTensor(not_leav),
                        "size": torch.LongTensor(size),
                    }
                    samples.append(sample)
                    vec, high, leav, not_leav, size = [], [], [], [], []

        # split samples into train, valid, test
        random.seed(123)
        random.shuffle(samples)
        self.n_samples = len(samples)
        counter = 1
        self.trainset, self.validset, self.testset = [], [], []
        for sample in tqdm(samples, desc="splitting samples"):
            if 1 <= counter <= 6:
                self.trainset.append(sample)
            elif 7 <= counter <= 8:
                self.validset.append(sample)
            else:
                self.testset.append(sample)
            counter += 1
            if counter > 10:
                counter = 1
        self.logger.info(f"validset:{len(self.validset)}, testset:{len(self.testset)}, trainset:{len(self.trainset)}")

    @ex.capture(prefix="model_setup")
    def init_model(self, checkpoint, input_dim, dense_dim, n_classes, bias):
        self.logger.info("model setup...")
        self.logger.info("\n" + pformat(ex.current_run.config["model_setup"]))

        self.reg_model = ProbingRegressionnNetwork(input_dim, dense_dim, bias)
        self.clf_model = ProbingClassificationNetwork(input_dim, dense_dim, n_classes, bias)
        self.logger.info("model constructed")
        if checkpoint:
            if Path(checkpoint).exists():
                self.model.load_state_dict(torch.load(checkpoint))
                self.logger.info(f"Loaded checkpoint model: {checkpoint}")
        self.logger.info("Done!\n")

    @ex.capture(prefix="training_setup")
    def train(self, with_cuda, cuda_device_id, max_epoch, random_seed, early_stopping_patience, learning_rate, model_dir):
        self.logger.info("training setup...")
        self.logger.info("\n" + pformat(ex.current_run.config["training_setup"]))

        # cuda
        if with_cuda:
            self.device = f"cuda:{cuda_device_id}"
        else:
            self.device = "cpu"

        # random seed
        torch.manual_seed(random_seed)
        random.seed(random_seed)

        # model validation
        def test_model(dataset, label_type, loss_func):
            self.model.eval()
            predicts = []
            expects = []
            MAE = 0
            with torch.no_grad():
                for sample in tqdm(dataset, desc="Testing model"):
                    labels = sample[label_type]
                    vec = sample["vec"]
                    if with_cuda:
                        vec = vec.to(self.device)
                        labels = labels.to(self.device)
                    # run model
                    out = self.model(vec)
                    out = out.squeeze()

                    # record test performance
                    if self.mode == "clf":
                        predicts.append(torch.max(out.data.cpu(), dim=1).indices)
                        expects.append(labels.data.cpu())
                    elif self.mode == "reg":
                        loss = loss_func(out, labels)
                        MAE += loss.item()

            if self.mode == "clf":
                predicts = np.array(torch.hstack(predicts))
                expects = np.array(torch.hstack(expects))
                metrics = accuracy(predicts, expects)
                metrics_type = "acc"
            elif self.mode == "reg":
                metrics = -MAE / len(dataset)
                metrics_type = "MAE"
            else:
                raise ValueError(f"Unsupported mode {self.mode}")
            return metrics, metrics_type

        def label_loop(label_type, mode):
            self.mode = mode

            # loss function & model
            if self.mode == "clf":
                loss_func = CrossEntropyLoss(reduction="mean")
                self.model = self.clf_model
            elif self.mode == "reg":
                loss_func = L1Loss(reduction="mean")
                self.model = self.reg_model
            else:
                raise ValueError(f"Unsupported mode: {self.mode}")
            if with_cuda:
                loss_func = loss_func.to(self.device)
                self.model = self.model.to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

            # early stop
            early_stopping = EarlyStopping(self.run_id, early_stopping_patience, self.logger)

            # training loop
            self.logger.info(f"Starting Training {label_type}\n")
            with Timing() as t_epoch:
                for epoch_id in range(max_epoch):
                    self.logger.info(f"<Starting epoch {epoch_id}>")

                    # Train
                    self.logger.info("[Training]")
                    self.model.train()
                    epoch_train_loss = 0
                    predicts = []
                    expects = []
                    MAE = 0
                    total = len(self.trainset)
                    with tqdm(total=total) as t_bar:
                        t_bar.set_description("Training")
                        for sample in self.trainset:
                            labels = sample[label_type]
                            vec = sample["vec"]
                            if with_cuda:
                                vec = vec.to(self.device)
                                labels = labels.to(self.device)

                            # run model
                            out = self.model(vec)
                            out = out.squeeze()

                            # calculate loss
                            loss = loss_func(out, labels)
                            epoch_train_loss += loss.item()

                            t_bar.update(1)
                            t_bar.set_description(f"Training, loss:{loss:.2f}")
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            # record train performance
                        if self.mode == "clf":
                            predicts.append(torch.max(out.data.cpu(), dim=1).indices)
                            expects.append(labels.data.cpu())
                        elif self.mode == "reg":
                            MAE += loss.item()

                    if self.mode == "clf":
                        predicts = np.array(torch.hstack(predicts))
                        expects = np.array(torch.hstack(expects))
                        metrics = accuracy(predicts, expects)
                        metrics_type = "acc"
                    elif self.mode == "reg":
                        metrics = -MAE / total
                        metrics_type = "MAE"

                    self.logger.info(
                        f"training end, took {t_epoch.measure():.2f} seconds,{label_type} {metrics_type}:{metrics:.4f}, current learning "
                        f"rate:{self.optimizer.param_groups[0]['lr']:.6f}"
                    )

                    # Validation
                    self.logger.info("[validation]")
                    metrics, metrics_type = test_model(self.validset, label_type, loss_func)
                    self.logger.info(f"{label_type} {metrics_type}:{metrics:.4f}")
                    # early stop
                    if not early_stopping.evaluate(metrics):
                        self.logger.info(f"Early stop!")
                        break
                    # checkpoint
                    if early_stopping.checkpoint:
                        self.model_name = f"model_{self.run_id}_{label_type}_{metrics_type}{early_stopping.best:.4f}.pt"
                        early_stopping.save_checkpoint(model_dir, self.model_name, self.model)

                    self.logger.info(
                        f"Epoch {epoch_id} end, avg batch loss: {epoch_train_loss / len(self.trainset)}, took {t_epoch.measure():.2f} seconds"
                    )
                    self.logger.info("-" * 80)

            # test
            self.logger.info("*" * 50)
            self.logger.info(f"Start testing...")
            with Timing() as t_test:
                self.model.load_state_dict(torch.load(str(Path(model_dir) / self.model_name)))
                metrics, metrics_type = test_model(self.testset, label_type, loss_func)
                self.logger.info(f"{label_type} {metrics_type}:{metrics:.4f}")
                self.logger.info(f"Test end, took {t_test.measure():.2f} seconds")
            self.logger.info("*" * 50)

        # train
        labels = {"height": "reg", "leaves": "clf", "not_leaves": "clf", "size": "clf"}
        # labels = {"leaves": "clf", "not_leaves": "clf", "size": "clf"}
        for label, mode in labels.items():
            label_loop(label, mode)
