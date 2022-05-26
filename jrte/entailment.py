import argparse
import json
import random
import re
import pandas as pd
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_metric
from pyknp import Juman
from sudachitra import ElectraSudachipyTokenizer
from sudachitra.tokenization_bert_sudachipy import BertSudachipyTokenizer
from transformers import (AutoConfig, BertConfig, AutoModelForSequenceClassification, BertForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments)



def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path", type=Path, default=Path("./data/jrte-corpus/data/")
    )
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--ga-steps", type=int, default=1)
    return parser.parse_args()


def seed_everything(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def compute_metrics(p):
    raw_pred, labels = p
    predictions = np.argmax(raw_pred, axis=1)
    
    result = {
        "macro_f1": load_metric("f1").compute(predictions=predictions, references=labels, average="macro")["f1"],
        "weighted_f1": load_metric("f1").compute(predictions=predictions, references=labels, average="weighted")["f1"],
        "accuracy": load_metric("accuracy").compute(predictions=predictions, references=labels)["accuracy"],
        "p=0": sum(predictions == 0),
        "p=1": sum(predictions == 1),
    }

    return result


def write_predictions(path, output, df_tr):
    predictions = np.argmax(output.predictions, axis=1)
    with open(path, "w") as fp:
        print("text_a\ttext_b\tprediction\tlabel", file=fp)
        for ta, tb, l, p in zip(df_tr["text_a"].values, df_tr["text_b"].values, df_tr["entailment"], predictions):
            print(f"{ta}\t{tb}\t{p}\t{l}", file=fp)


class EntailmentDataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def tokenize_func(tokenizer, text_a, text_b, options, juman=None):
    if juman:
        def split(x):
            result = juman.analysis(x)
            text = " ".join([mrph.midasi for mrph in result.mrph_list()])
            return text
        text_a = [split(t) for t in text_a]
        text_b = [split(t) for t in text_b]
    return tokenizer(text_a, text_pair=text_b, **options)


def main():
    opts = get_opts()

    seed_everything(42)

    model_name = opts.model_name

    if (
        model_name == "megagonlabs/transformers-ud-japanese-electra-base-discriminator"
        or model_name.endswith("electra-base-japanese-mc4")
    ):
        tokenizer = ElectraSudachipyTokenizer.from_pretrained(model_name)
    elif model_name.endswith("chiTra-1.0"):
        tokenizer = BertSudachipyTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_name in (
        "nlp-waseda/roberta-base-japanese",
        "nlp-waseda/roberta-large-japanese"
    ):
        juman = Juman()
    else:
        juman = None

    if model_name.endswith("chiTra-1.0"):
        config = BertConfig.from_pretrained(model_name, num_labels=2)
        model = BertForSequenceClassification.from_pretrained(model_name, config=config)
    else:
        config = AutoConfig.from_pretrained(model_name, num_labels=2)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    param = f"maxlen_{opts.max_len}_bs_{opts.bs}_lr_{opts.lr}_epoch{opts.epochs}_warmup{opts.warmup_ratio}_gas{opts.ga_steps}"
    model_output_dir = (
        Path(__file__).parent.resolve()
        / f"models/entailment/{model_name}/{param}"
    )
    result_dir = (
        Path(__file__).parent.resolve()
        / f"results/entailment/{model_name}/{param}"
    )
    result_dir.mkdir(exist_ok=True, parents=True)

    dfs = [
        pd.read_csv(p, sep="\t", header=None) for p in opts.data_path.glob("rte.*")
    ]
    df = pd.concat(dfs, axis=0).reset_index(drop=True)
    df.columns = ["id", "entailment", "text_a", "text_b", "judge", "reason", "split"]
    df_tr = df[df["split"] == "train"]
    df_val = df[df["split"] == "dev"]
    df_te = df[df["split"] == "test"]

    tokenize_options = dict(
        padding="max_length",
        truncation=True,
        max_length=opts.max_len
    )
    train_dataset = EntailmentDataset(
        encodings=tokenize_func(tokenizer, df_tr["text_a"].values.tolist(), df_tr["text_b"].values.tolist(), tokenize_options, juman),
        labels=df_tr["entailment"].values.tolist()
    )
    valid_dataset = EntailmentDataset(
        encodings=tokenize_func(tokenizer, df_val["text_a"].values.tolist(), df_val["text_b"].values.tolist(), tokenize_options, juman),
        labels=df_val["entailment"].values.tolist()
    )
    eval_dataset = EntailmentDataset(
        encodings=tokenize_func(tokenizer, df_te["text_a"].values.tolist(), df_te["text_b"].values.tolist(), tokenize_options, juman),
        labels=df_te["entailment"].values.tolist()
    )

    train_args = TrainingArguments(
        output_dir=model_output_dir,
        do_train=True,
        do_eval=True,
        do_predict=True,
        per_device_train_batch_size=opts.bs,
        per_device_eval_batch_size=opts.bs,
        learning_rate=opts.lr,
        warmup_ratio=opts.warmup_ratio,
        gradient_accumulation_steps=opts.ga_steps,
        num_train_epochs=opts.epochs,
        evaluation_strategy="epoch",
        # save_strategy="no",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate()

    output = trainer.predict(eval_dataset)
    print(output.metrics)

    with open(result_dir / "test_metrics.json", "w") as fp:
        json.dump(output.metrics, fp)

    write_predictions(
        result_dir / "test_predictions.tsv",
        output, df_te
    )

    trainer.save_model(model_output_dir)


if __name__ == "__main__":
    main()
