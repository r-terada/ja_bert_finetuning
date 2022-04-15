import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_metric
from pyknp import Juman
from seqeval.metrics import classification_report
from sklearn.model_selection import train_test_split
from sudachitra import ElectraSudachipyTokenizer
from transformers import (AutoConfig, AutoModelForTokenClassification,
                          AutoTokenizer, EarlyStoppingCallback, Trainer,
                          TrainingArguments)


def get_opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json-path", type=Path, default=Path("./data/ner-wikipedia-dataset/ner.json")
    )
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    return parser.parse_args()


def seed_everything(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


metric = load_metric("seqeval")


def fn_compute_metrics(id2label):
    def fn(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        y_pred = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        y_true = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        results = metric.compute(predictions=y_pred, references=y_true)

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    return fn


def write_predictions(path, output, id2label, eval_dataset):
    predictions = np.argmax(output.predictions, axis=2)

    y_pred = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, output.label_ids)
    ]
    with open(path, "w") as fp:
        print("token\tprediction\tlabel", file=fp)
        for data, pred in zip(eval_dataset, y_pred):
            for p, t, l in zip(pred, data["tokens"], data["labels"]):
                print(f"{t}\t{p}\t{l}", file=fp)
            print(file=fp)


def write_classification_report(path, output, id2label):
    predictions = np.argmax(output.predictions, axis=2)

    y_pred = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, output.label_ids)
    ]
    y_true = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, output.label_ids)
    ]

    with open(path, "w") as fp:
        print(
            classification_report(y_true, y_pred),
            file=fp,
        )


@dataclass
class BIOData:
    tokens: List[str]
    labels: List[str]


class NERDataset:
    def __init__(
        self,
        tokenizer,
        json_path,
        valid_size=0.1,
        test_size=0.2,
        subword_prefix="##",
        use_juman=False,
        ws_is_removed=True,
    ):
        self.tokenizer = tokenizer
        self.subword_prefix = subword_prefix
        self.use_juman = use_juman
        self.ws_is_removed = ws_is_removed
        dataset = self._load_dataset(json_path)
        self.label2id, self.id2label = self._create_label_vocab(dataset)
        _tr, ev = train_test_split(dataset, test_size=test_size, random_state=42)
        tr, val = train_test_split(_tr, test_size=valid_size, random_state=42)
        self.train_dataset = [asdict(d) for d in tr]
        self.valid_dataset = [asdict(d) for d in val]
        self.eval_dataset = [asdict(d) for d in ev]

    def save(self, base_dir):
        base_dir.mkdir(parents=True, exist_ok=True)

        def _save(fname, dataset):
            with open(base_dir / fname, "w") as fp:
                for d in dataset:
                    for t, l in zip(d["tokens"], d["labels"]):
                        print(f"{t}\t{l}", file=fp)
                    print(file=fp)

        _save("ner.train", self.train_dataset)
        _save("ner.valid", self.valid_dataset)
        _save("ner.eval", self.eval_dataset)

    def _create_label_vocab(
        self, dataset: list
    ) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
        labels = [d.labels for d in dataset]
        unique_labels = list(set(sum(labels, [])))
        label2id = {}
        for i, label in enumerate(unique_labels):
            label2id[label] = i
        id2label = {v: k for k, v in label2id.items()}
        return label2id, id2label

    def _to_bio_labels(
        self,
        tokens: List[str],
        entities: Tuple[List[int], str],
        ws_pos: List[re.Match],
        subword_prefix: Optional[str] = "##",
    ) -> List[str]:
        labels = ["O"] * len(tokens)

        for ent in entities:
            begin = ent["span"][0]
            end = ent["span"][1]
            label = ent["type"]
            position = 0
            targets = []
            ws_i = 0

            for i, token in enumerate(tokens):
                if ws_i < len(ws_pos):
                    if ws_pos[ws_i].span()[0] <= position < ws_pos[ws_i].span()[1]:
                        position += ws_pos[ws_i].span()[1] - ws_pos[ws_i].span()[0]
                        ws_i += 1
                if token.startswith(subword_prefix):
                    token = re.sub(f"^{subword_prefix}", "", token)
                if begin <= position < end:
                    targets.append(i)
                position += len(token)

            if targets:
                labels[targets[0]] = f"B-{label}"
                for t in targets[1:]:
                    labels[t] = f"I-{label}"

        return labels

    def _load_dataset(self, json_path):
        with open(json_path, "r") as i_fp:
            units = json.load(i_fp)

        if self.use_juman:
            jumanpp = Juman()

        dataset = []
        for unit in units:
            if self.use_juman:
                result = jumanpp.analysis(unit["text"])
                tokens = [mrph.midasi.lstrip("\\") for mrph in result.mrph_list()]
            else:
                tokens = self.tokenizer.tokenize(unit["text"])

            if hasattr(self.tokenizer, "subword_tokenizer"):
                _new_tokens = []
                for token in tokens:
                    _new_tokens.extend(self.tokenizer.subword_tokenizer.tokenize(token))
                tokens = _new_tokens

            if self.ws_is_removed:
                ws_pos = list(re.finditer("\s+", unit["text"]))
            else:
                ws_pos = []

            labels = self._to_bio_labels(
                tokens, unit["entities"], ws_pos, self.subword_prefix
            )
            dataset.append(BIOData(tokens, labels))
        return dataset


def main():
    opts = get_opts()

    seed_everything(42)

    model_name = opts.model_name
    # model_name = "megagonlabs/transformers-ud-japanese-electra-base-discriminator"
    # model_name = "izumi-lab/electra-base-japanese-discriminator"
    # model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
    # model_name = "cl-tohoku/bert-base-japanese-v2"
    # model_name = "cl-tohoku/bert-large-japanese"
    # model_name = "nlp-waseda/roberta-base-japanese"

    if model_name == "megagonlabs/transformers-ud-japanese-electra-base-discriminator":
        tokenizer = ElectraSudachipyTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_name == "nlp-waseda/roberta-base-japanese":
        dataset = NERDataset(
            tokenizer, opts.json_path, use_juman=True, ws_is_removed=False
        )
    else:
        dataset = NERDataset(tokenizer, opts.json_path)

    cache_path = Path(__file__).parent.resolve() / f"cache/{model_name}"
    dataset.save(cache_path)

    config = AutoConfig.from_pretrained(
        model_name,
        label2id=dataset.label2id,
        id2label=dataset.id2label,
    )
    model = AutoModelForTokenClassification.from_pretrained(model_name, config=config)

    model_output_dir = (
        Path(__file__).parent.resolve()
        / f"models/{model_name}/maxlen_{opts.max_len}_bs_{opts.bs}_lr_{opts.lr}_epoch{opts.epochs}"
    )

    train_args = TrainingArguments(
        output_dir=model_output_dir,
        do_train=True,
        do_eval=True,
        do_predict=True,
        per_device_train_batch_size=opts.bs,
        per_device_eval_batch_size=opts.bs,
        learning_rate=opts.lr,
        num_train_epochs=opts.epochs,
        evaluation_strategy="epoch",
        # save_strategy="no",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        debug="underflow_overflow"
    )

    def data_collator(features: list) -> dict:
        # x = [tokenizer.convert_tokens_to_string(f["tokens"]) for f in features]
        x = [f["tokens"] for f in features]
        y = [f["labels"] for f in features]
        inputs = tokenizer(
            x,
            return_tensors=None,
            padding="max_length",
            truncation=True,
            max_length=opts.max_len,
            # FIXME: is_split_into_words with sudachipy convert ##hoge to # # hoge
            is_split_into_words=True,
        )
        input_labels = []
        for labels in y:
            pad_list = [-100] * opts.max_len
            for i, label in enumerate(labels[: opts.max_len - 2]):
                # 0-th token is [CLS]
                pad_list.insert(i + 1, dataset.label2id[label])
            input_labels.append(pad_list[: opts.max_len])
        inputs["labels"] = input_labels
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in inputs.items()}
        return batch

    compute_metrics = fn_compute_metrics(dataset.id2label)

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.valid_dataset,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate()

    output = trainer.predict(dataset.eval_dataset)
    print(output.metrics)

    result_dir = (
        Path(__file__).parent.resolve()
        / f"results/{model_name}/maxlen_{opts.max_len}_bs_{opts.bs}_lr_{opts.lr}_epoch{opts.epochs}"
    )
    result_dir.mkdir(exist_ok=True, parents=True)

    with open(result_dir / "test_metrics.json", "w") as fp:
        json.dump(output.metrics, fp)

    write_predictions(
        result_dir / "test_predictions.txt",
        output,
        dataset.id2label,
        dataset.eval_dataset,
    )

    write_classification_report(
        result_dir / "classification_report.txt",
        output,
        dataset.id2label,
    )

    trainer.save_model(model_output_dir)


if __name__ == "__main__":
    main()
