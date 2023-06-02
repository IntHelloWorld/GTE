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
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
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

import numpy as np
import torch
from model import GCBModel
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import WEIGHTS_NAME, AdamW, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, get_linear_schedule_with_warmup

from dataset.read_dataset_gcb import GCBdataset

cpu_cont = 16
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """Train the model"""

    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4)

    max_steps = args.epochs * len(train_dataloader)
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * max_steps, num_training_steps=max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  batch size = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", max_steps)
    best_acc = 0.0
    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        losses = []
        for step, batch in enumerate(bar):
            (inputs_ids, position_idx, attn_mask, labels) = [x.to(args.device) for x in batch]
            model.train()
            loss, logits = model(inputs_ids, position_idx, attn_mask, labels)

            if args.n_gpu > 1:
                loss = loss.mean()

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
        if results["eval_acc"] >= best_acc:
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


def evaluate(args, model, tokenizer, eval_when_training=False):
    # build dataloader
    eval_dataset = GCBdataset(args, args.eval_data_file, tokenizer)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        (inputs_ids, position_idx, attn_mask, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(inputs_ids, position_idx, attn_mask, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    preds = logits.argmax(-1)
    eval_acc = np.mean(y_trues == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
    }
    return result


def test(args, model, tokenizer):
    # build dataloader
    eval_dataset = GCBdataset(args, args.test_data_file, tokenizer)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        inputs_ids, position_idx, attn_mask, labels = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(inputs_ids, position_idx, attn_mask, labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # output result
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    preds = logits.argmax(-1)
    test_acc = np.mean(y_trues == preds)
    # with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
    #     for example, pred in zip(eval_dataset.examples, preds):
    #         f.write(str(pred) + "\n")
    logger.info("Acc = %s", str(round(test_acc, 4)))


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default="/home/qyh/Desktop/github/dataset/CodeNet/Project_CodeNet_Python800_RATIO6-2-2",
        type=str,
        help="The input training data directory.",
    )
    parser.add_argument("--output_dir", default="GraphCodeBERT_output/Python800", type=str, help="The output directory.")
    parser.add_argument("--logs_dir", default="logs/Python800", type=str, help="Dir for logs.")
    parser.add_argument("--lang", default="java", type=str, help="Language.")

    ## Other parameters
    parser.add_argument("--cache_dir", default="cache", type=str, help="Dir for store downloaded sources.")
    parser.add_argument("--code_length", default=512, type=int, help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--data_flow_length", default=64, type=int, help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--do_train", default=True, help="Whether to run training.")
    parser.add_argument("--do_eval", default=False, help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", default=True, help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=5, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--epochs", type=int, default=50, help="training epochs")

    args = parser.parse_args()
    args.train_data_file = args.data_dir + "/train"
    args.eval_data_file = args.data_dir + "/valid"
    if args.do_test:
        args.test_data_file = args.data_dir + "/test"

    # Setup CUDA, GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1
    args.device = device

    # Setup logging
    run_id = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    logging.basicConfig(
        filename=os.path.join(args.logs_dir, f"GraphCodeBERT_{run_id}.log"),
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)
    base = "microsoft/graphcodebert-base"
    config = RobertaConfig.from_pretrained(base, cache_dir=args.cache_dir)
    config.num_labels = 800
    tokenizer = RobertaTokenizer.from_pretrained(base, cache_dir=args.cache_dir)
    model = RobertaForSequenceClassification.from_pretrained(base, config=config, cache_dir=args.cache_dir)

    model = GCBModel(model, config, tokenizer, args)
    logger.info("Training/evaluation parameters %s", args)

    # Load checkpoint
    # checkpoint = "/home/qyh/projects/GTE/reproduce/Pretrained_Models/GraphCodeBERT_output/checkpoint-best-acc/model.bin"
    # model.load_state_dict(torch.load(checkpoint))

    # Training
    if args.do_train:
        train_dataset = GCBdataset(args, args.train_data_file, tokenizer)
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = "checkpoint-best-acc/model.bin"
        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer)

    if args.do_test:
        checkpoint_prefix = "checkpoint-best-acc/model.bin"
        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, tokenizer)

    return results


if __name__ == "__main__":
    main()
