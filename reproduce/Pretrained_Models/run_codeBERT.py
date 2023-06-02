# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import json
import logging
import multiprocessing
import os
import pickle
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from model import CBModel
from torch import optim
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import WEIGHTS_NAME, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, get_linear_schedule_with_warmup

from dataset.read_dataset_cb import CodeDataset

os.chdir(Path(__file__).parent)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer):
    """Train the model"""
    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=8, pin_memory=True)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    max_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1, num_training_steps=max_steps)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", max_steps)
    best_acc = 0.0
    model.zero_grad()

    for idx in range(args.num_train_epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        losses = []
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()
            loss, logits = model(inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            losses.append(loss.item())
            bar.set_description("epoch {} loss {}".format(idx, round(np.mean(losses), 3)))
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        logger.info(f"current lr:{optimizer.param_groups[0]['lr']} ")
        results = evaluate(args, model, tokenizer)
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value, 4))

        # Save model checkpoint
        if results["eval_acc"] > best_acc:
            best_acc = results["eval_acc"]
            logger.info("  " + "*" * 20)
            logger.info("  Best acc:%s", round(best_acc, 4))
            logger.info("  " + "*" * 20)

            checkpoint_prefix = "checkpoint-best-acc"
            output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, "module") else model
            output_dir = os.path.join(output_dir, "{}".format("model.bin"))
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer):
    eval_output_dir = args.output_dir

    eval_dataset = CodeDataset(args, args.eval_data_file, tokenizer)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits.argmax(-1)
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
    }
    return result


def test(args, model, tokenizer):
    # Note that DistributedSampler samples randomly
    eval_dataset = CodeDataset(args, args.test_data_file, tokenizer)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            logit = model(inputs)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits.argmax(-1)
    test_acc = np.mean(labels == preds)
    with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
        for example, pred in zip(eval_dataset.data, preds):
            f.write(str(pred) + "\n")
    logger.info("Acc = %s", str(round(test_acc, 4)))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default="/home/qyh/Desktop/github/dataset/CodeNet/Project_CodeNet_C++1400_RATIO6-2-2",
        type=str,
        help="The input training data directory.",
    )
    parser.add_argument("--output_dir", default="CodeBERT_output/C++1400", type=str, help="The output directory.")
    parser.add_argument("--logs_dir", default="logs/C++1400", type=str, help="Dir for logs.")

    ## Other parameters
    parser.add_argument("--cache_dir", default="cache", type=str, help="Dir for store downloaded sources.")
    parser.add_argument("--block_size", default=512, type=int, help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", default=True, help="Whether to run training.")
    parser.add_argument("--do_eval", default=False, help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=True, help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=5, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--seed", type=int, default=123456, help="random seed for initialization")
    parser.add_argument("--num_train_epochs", type=int, default=50, help="num_train_epochs")

    args = parser.parse_args()
    args.train_data_file = args.data_dir + "/train"
    args.eval_data_file = args.data_dir + "/valid"
    if args.do_test:
        args.test_data_file = args.data_dir + "/test"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.n_gpu = torch.cuda.device_count()
    args.n_gpu = 1
    args.device = device

    # Setup logging
    run_id = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    logging.basicConfig(
        filename=os.path.join(args.logs_dir, f"codeBERT_{run_id}.log"),
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model
    base = "microsoft/codebert-base"
    config = RobertaConfig.from_pretrained(base, cache_dir=args.cache_dir)
    config.num_labels = 1400
    tokenizer = RobertaTokenizer.from_pretrained(base, cache_dir=args.cache_dir)
    model = RobertaForSequenceClassification.from_pretrained(base, config=config, cache_dir=args.cache_dir)
    model = CBModel(model, config, tokenizer, args)

    # Load checkpoint
    # checkpoint = "/home/qyh/projects/GTE/reproduce/Pretrained_Models/output/checkpoint-best-acc/model.bin"
    # model.load_state_dict(torch.load(checkpoint))

    # multi-gpu training (should be after apex fp16 initialization)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = CodeDataset(args, args.train_data_file, tokenizer)
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = "checkpoint-best-acc/model.bin"
        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test:
        checkpoint_prefix = "checkpoint-best-acc/model.bin"
        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, tokenizer)

    return results


if __name__ == "__main__":
    main()
