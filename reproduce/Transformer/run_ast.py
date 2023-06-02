from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import pickle
import random

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from model import TransformerClf
from torch import optim
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
)

from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from read_dataset_ast import CodeDataset

os.chdir(Path(__file__).parent)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, vocab):
    """Train the model"""
    train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
    eval_dataset = CodeDataset(args, args.eval_data_file, vocab)

    # Prepare optimizer and schedule (linear warmup and decay)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    max_steps = args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * max_steps, num_training_steps=max_steps)

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

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            losses.append(loss.item())
            bar.set_description("epoch {} loss {}".format(idx, round(np.mean(losses), 3)))
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()

        logger.info(f"current lr:{optimizer.param_groups[0]['lr']} ")
        results = evaluate(args, model, eval_dataset)
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


def evaluate(args, model, eval_dataset):
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


def test(args, model, vocab):
    # Note that DistributedSampler samples randomly
    eval_dataset = CodeDataset(args, args.test_data_file, vocab)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
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
    os.chdir(str(Path(__file__).parent))

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default="/home/qyh/Desktop/github/dataset/CodeNet/Project_CodeNet_Java250_RATIO6-2-2",
        type=str,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--vocab_dir",
        default="/home/qyh/projects/GTE/vocabulary/token_to_index_PC_Java250.pkl",
        type=str,
        help="The vocabulary file.",
    )
    parser.add_argument("--so_file", default="/home/qyh/projects/GTE/utils/python-java-c-cpp-languages.so", type=str, help="The .so file.")
    parser.add_argument("--lang", default="java", type=str, help="The language.")
    parser.add_argument(
        "--output_dir",
        default="./output",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    ## Other parameters
    parser.add_argument("--block_size", default=512, type=int, help="Optional input sequence length after tokenization.")
    parser.add_argument("--do_train", default=True, help="Whether to run training.")
    parser.add_argument("--do_test", default=True, help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=64, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=0.00005, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--seed", type=int, default=123456, help="random seed for initialization")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="num_train_epochs")

    # model parameters
    parser.add_argument("--n_layers", default=4, type=int)
    parser.add_argument("--n_head", default=8, type=int)
    parser.add_argument("--n_ff", default=1024, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--n_label", default=250, type=int)
    parser.add_argument("--d_model", default=512, type=int)
    parser.add_argument("--ckpt", default=None, type=str)

    args = parser.parse_args()
    args.train_data_file = args.data_dir + "/train"
    args.eval_data_file = args.data_dir + "/valid"
    if args.do_test:
        args.test_data_file = args.data_dir + "/test"

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    args.device = device
    run_id = datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    run_name = "ast_record_time_Java250"

    # create dir
    args.output_dir = os.path.join(args.output_dir, run_name, run_id)
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True)
    logs_dir = os.path.join("logs", run_name)
    if not Path(logs_dir).exists():
        Path(logs_dir).mkdir()

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(logs_dir, f"Transformer_{run_id}.log"),
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning("device: %s", device)

    # Set seed
    set_seed(args.seed)

    # model
    vocab = pickle.load(open(args.vocab_dir, "rb"))
    args.vocab_size = len(vocab)
    logger.info(f"vocabulary loaded, size:{args.vocab_size}")
    model = TransformerClf(args.d_model, args.n_head, args.n_layers, args.dropout, args.vocab_size, args.n_label)
    model.to(args.device)

    # Load checkpoint
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt))

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = CodeDataset(args, args.train_data_file, vocab)
        train(args, train_dataset, model, vocab)

    # Test
    if args.do_test:
        checkpoint_prefix = "checkpoint-best-acc/model.bin"
        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, vocab)


if __name__ == "__main__":
    main()
