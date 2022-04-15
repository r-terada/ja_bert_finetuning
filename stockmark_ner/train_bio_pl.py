import argparse
import re
from pathlib import Path
import itertools
import random
import json
from tqdm import tqdm
import numpy as np
import unicodedata

import torch
from pyknp import Juman
from torch.utils.data import DataLoader
from sudachitra import ElectraSudachipyTokenizer
from transformers import AutoTokenizer, AutoModelForTokenClassification
import pytorch_lightning as pl


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


def normalize(s):
    return unicodedata.normalize("NFKC",s)


class JumanWordTokenizer:

    def __init__(self):
        self.jumanpp = Juman()

    def tokenize(self, text):
        result = self.jumanpp.analysis(text)
        tokens = [mrph.midasi.lstrip("\\") for mrph in result.mrph_list()]
        return tokens


class NER_tokenizer_BIO:

    # 初期化時に固有表現のカテゴリーの数`num_entity_type`を
    # 受け入れるようにする。
    def __init__(self, tokenizer, num_entity_type, word_tokenizer=None, subword_prefix="##"):
        self.tokenizer = tokenizer
        self.num_entity_type = num_entity_type
        self.subword_prefix = subword_prefix
        if word_tokenizer:
            self.word_tokenizer = word_tokenizer
        else:
            if hasattr(tokenizer, "word_tokenizer"):
                self.word_tokenizer = tokenizer.word_tokenizer
            else:
                self.word_tokenizer = None
        if hasattr(tokenizer, "subword_tokenizer"):
            self.subword_tokenizer = tokenizer.subword_tokenizer
        else:
            self.subword_tokenizer = tokenizer

    def encode_plus_tagged(self, text, entities, max_length):
        """
        文章とそれに含まれる固有表現が与えられた時に、
        符号化とラベル列の作成を行う。
        """
        # 固有表現の前後でtextを分割し、それぞれのラベルをつけておく。
        splitted = [] # 分割後の文字列を追加していく
        position = 0
        for entity in entities:
            start = entity['span'][0]
            end = entity['span'][1]
            label = entity['type_id']
            splitted.append({'text':text[position:start], 'label':0})
            splitted.append({'text':text[start:end], 'label':label})
            position = end
        splitted.append({'text': text[position:], 'label':0})
        splitted = [ s for s in splitted if s['text'] ]

        # 分割されたそれぞれの文章をトークン化し、ラベルをつける。
        tokens = [] # トークンを追加していく
        labels = [] # ラベルを追加していく
        for s in splitted:
            tokens_splitted = self.tokenizer.tokenize(s['text'])
            label = s['label']
            if label > 0: # 固有表現
                # まずトークン全てにI-タグを付与
                labels_splitted =  \
                    [ label + self.num_entity_type ] * len(tokens_splitted)
                # 先頭のトークンをB-タグにする
                labels_splitted[0] = label
            else: # それ以外
                labels_splitted =  [0] * len(tokens_splitted)
            
            tokens.extend(tokens_splitted)
            labels.extend(labels_splitted)

        # 符号化を行いBERTに入力できる形式にする。
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        encoding = self.tokenizer.prepare_for_model(
            input_ids, 
            max_length=max_length, 
            padding='max_length',
            truncation=True
        ) 

        # ラベルに特殊トークンを追加
        labels = [0] + labels[:max_length-2] + [0]
        labels = labels + [0]*( max_length - len(labels) )
        encoding['labels'] = labels

        return encoding

    def encode_plus_untagged(
        self, text, max_length=None, return_tensors=None
    ):
        """
        文章をトークン化し、それぞれのトークンの文章中の位置も特定しておく。
        IO法のトークナイザのencode_plus_untaggedと同じ
        """
        # 文章のトークン化を行い、
        # それぞれのトークンと文章中の文字列を対応づける。
        tokens = [] # トークンを追加していく。
        tokens_original = [] # トークンに対応する文章中の文字列を追加していく。
        if self.word_tokenizer:
            words = self.word_tokenizer.tokenize(text) # MeCabで単語に分割
        else:
            words = [text]
        for word in words:
            # 単語をサブワードに分割
            tokens_word = self.subword_tokenizer.tokenize(word) 
            tokens.extend(tokens_word)
            if tokens_word[0] == '[UNK]': # 未知語への対応
                tokens_original.append(word)
            else:
                tokens_original.extend([
                    re.sub(f"^{self.subword_prefix}", "", token) for token in tokens_word
                ])

        # 各トークンの文章中での位置を調べる。（空白の位置を考慮する）
        position = 0
        spans = [] # トークンの位置を追加していく。
        for token in tokens_original:
            l = len(token)
            while 1:
                if token != text[position:position+l]:
                    position += 1
                else:
                    spans.append([position, position+l])
                    position += l
                    break
        
        # 符号化を行いBERTに入力できる形式にする。
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
        encoding = self.tokenizer.prepare_for_model(
            input_ids, 
            max_length=max_length, 
            padding='max_length' if max_length else False, 
            truncation=True if max_length else False
        )
        sequence_length = len(encoding['input_ids'])
        # 特殊トークン[CLS]に対するダミーのspanを追加。
        spans = [[-1, -1]] + spans[:sequence_length-2] 
        # 特殊トークン[SEP]、[PAD]に対するダミーのspanを追加。
        spans = spans + [[-1, -1]] * ( sequence_length - len(spans) ) 

        # 必要に応じてtorch.Tensorにする。
        if return_tensors == 'pt':
            encoding = { k: torch.tensor([v]) for k, v in encoding.items() }

        return encoding, spans

    @staticmethod
    def Viterbi(scores_bert, num_entity_type, penalty=10000):
        """
        Viterbiアルゴリズムで最適解を求める。
        """
        m = 2*num_entity_type + 1
        penalty_matrix = np.zeros([m, m])
        for i in range(m):
            for j in range(1+num_entity_type, m):
                if not ( (i == j) or (i+num_entity_type == j) ): 
                    penalty_matrix[i,j] = penalty
        
        path = [ [i] for i in range(m) ]
        scores_path = scores_bert[0] - penalty_matrix[0,:]
        scores_bert = scores_bert[1:]

        for scores in scores_bert:
            assert len(scores) == 2*num_entity_type + 1
            score_matrix = np.array(scores_path).reshape(-1,1) \
                + np.array(scores).reshape(1,-1) \
                - penalty_matrix
            scores_path = score_matrix.max(axis=0)
            argmax = score_matrix.argmax(axis=0)
            path_new = []
            for i, idx in enumerate(argmax):
                path_new.append( path[idx] + [i] )
            path = path_new

        labels_optimal = path[np.argmax(scores_path)]
        return labels_optimal

    def convert_bert_output_to_entities(self, text, scores, spans):
        """
        文章、分類スコア、各トークンの位置から固有表現を得る。
        分類スコアはサイズが（系列長、ラベル数）の2次元配列
        """
        assert len(spans) == len(scores)
        num_entity_type = self.num_entity_type
        
        # 特殊トークンに対応する部分を取り除く
        scores = [score for score, span in zip(scores, spans) if span[0]!=-1]
        spans = [span for span in spans if span[0]!=-1]

        # Viterbiアルゴリズムでラベルの予測値を決める。
        labels = self.Viterbi(scores, num_entity_type)

        # 同じラベルが連続するトークンをまとめて、固有表現を抽出する。
        entities = []
        for label, group \
            in itertools.groupby(enumerate(labels), key=lambda x: x[1]):
            
            group = list(group)
            start = spans[group[0][0]][0]
            end = spans[group[-1][0]][1]

            if label != 0: # 固有表現であれば
                if 1 <= label <= num_entity_type:
                     # ラベルが`B-`ならば、新しいentityを追加
                    entity = {
                        "name": text[start:end],
                        "span": [start, end],
                        "type_id": label
                    }
                    entities.append(entity)
                else:
                    # ラベルが`I-`ならば、直近のentityを更新
                    entity['span'][1] = end 
                    entity['name'] = text[entity['span'][0]:entity['span'][1]]
                
        return entities


class BertForTokenClassification_pl(pl.LightningModule):
        
    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters()
        self.bert_tc = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
    def training_step(self, batch, batch_idx):
        output = self.bert_tc(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        output = self.bert_tc(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def create_dataset(tokenizer, dataset, max_length):
    """
    データセットをデータローダに入力できる形に整形。
    """
    dataset_for_loader = []
    for sample in dataset:
        text = sample['text']
        entities = sample['entities']
        encoding = tokenizer.encode_plus_tagged(
            text, entities, max_length=max_length
        )
        encoding = { k: torch.tensor(v) for k, v in encoding.items() }
        dataset_for_loader.append(encoding)
    return dataset_for_loader


def evaluate_model(entities_list, entities_predicted_list, type_id=None):
    """
    正解と予測を比較し、モデルの固有表現抽出の性能を評価する。
    type_idがNoneのときは、全ての固有表現のタイプに対して評価する。
    type_idが整数を指定すると、その固有表現のタイプのIDに対して評価を行う。
    """
    num_entities = 0 # 固有表現(正解)の個数
    num_predictions = 0 # BERTにより予測された固有表現の個数
    num_correct = 0 # BERTにより予測のうち正解であった固有表現の数

    # それぞれの文章で予測と正解を比較。
    # 予測は文章中の位置とタイプIDが一致すれば正解とみなす。
    for entities, entities_predicted \
        in zip(entities_list, entities_predicted_list):

        if type_id:
            entities = [ e for e in entities if e['type_id'] == type_id ]
            entities_predicted = [ 
                e for e in entities_predicted if e['type_id'] == type_id
            ]
            
        get_span_type = lambda e: (e['span'][0], e['span'][1], e['type_id'])
        set_entities = set( get_span_type(e) for e in entities )
        set_entities_predicted = \
            set( get_span_type(e) for e in entities_predicted )

        num_entities += len(entities)
        num_predictions += len(entities_predicted)
        num_correct += len( set_entities & set_entities_predicted )

    # 指標を計算
    precision = num_correct/num_predictions # 適合率
    recall = num_correct/num_entities # 再現率
    f_value = 2*precision*recall/(precision+recall) # F値

    result = {
        'num_entities': num_entities,
        'num_predictions': num_predictions,
        'num_correct': num_correct,
        'precision': precision,
        'recall': recall,
        'f_value': f_value
    }

    return result


def main():

    opts = get_opts()

    pl.seed_everything(42)

    base_dir = Path(__file__).parent.resolve()

    model_tokenizer = AutoTokenizer.from_pretrained(opts.model_name)

    if opts.model_name == "nlp-waseda/roberta-base-japanese":
        tokenizer = NER_tokenizer_BIO(
            model_tokenizer,
            num_entity_type=8,
            word_tokenizer=JumanWordTokenizer(),
            subword_prefix="▁"
        )
    else:
        tokenizer = NER_tokenizer_BIO(
            model_tokenizer,
            num_entity_type=8,
        )

    # データセットの作成
    dataset = json.load(open(opts.json_path,'r'))

    type_id_dict = {
        "人名": 1,
        "法人名": 2,
        "政治的組織名": 3,
        "その他の組織名": 4,
        "地名": 5,
        "施設名": 6,
        "製品名": 7,
        "イベント名": 8
    }

    # カテゴリーをラベルに変更、文字列の正規化する。
    for sample in dataset:
        sample['text'] = unicodedata.normalize('NFKC', sample['text'])
        for e in sample["entities"]:
            e['type_id'] = type_id_dict[e['type']]
            del e['type']

    random.shuffle(dataset)
    n = len(dataset)
    n_train = int(n*0.6)
    n_val = int(n*0.2)
    dataset_train = dataset[:n_train]
    dataset_val = dataset[n_train:n_train+n_val]
    dataset_test = dataset[n_train+n_val:]
    max_length = opts.max_len
    dataset_train_for_loader = create_dataset(
        tokenizer, dataset_train, max_length
    )
    dataset_val_for_loader = create_dataset(
        tokenizer, dataset_val, max_length
    )

    # データローダの作成
    dataloader_train = DataLoader(
        dataset_train_for_loader, batch_size=opts.bs, shuffle=True
    )
    dataloader_val = DataLoader(dataset_val_for_loader, batch_size=256)

    # ファインチューニング
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_weights_only=True,
        dirpath='./model_pl_BIO/'
    )

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=opts.epochs,
        callbacks=[checkpoint]
    )

    num_entity_type = 8
    num_labels = 2 * num_entity_type + 1
    model = BertForTokenClassification_pl(
        opts.model_name, num_labels=num_labels, lr=opts.lr
    )

    trainer.fit(model, dataloader_train, dataloader_val)
    best_model_path = checkpoint.best_model_path

    # 性能評価
    model = BertForTokenClassification_pl.load_from_checkpoint(
        best_model_path
    ) 
    bert_tc = model.bert_tc.cuda()

    entities_list = [] # 正解の固有表現を追加していく
    entities_predicted_list = [] # 抽出された固有表現を追加していく
    for sample in tqdm(dataset_test, desc="test"):
        text = sample['text']
        encoding, spans = tokenizer.encode_plus_untagged(
            text, return_tensors='pt'
        )
        encoding = { k: v.cuda() for k, v in encoding.items() } 
        
        with torch.no_grad():
            output = bert_tc(**encoding)
            scores = output.logits
            scores = scores[0].cpu().numpy().tolist()
            
        # 分類スコアを固有表現に変換する
        entities_predicted = tokenizer.convert_bert_output_to_entities(
            text, scores, spans
        )

        entities_list.append(sample['entities'])
        entities_predicted_list.append( entities_predicted )

    result_dir = base_dir / "results_pl" / opts.model_name
    result_dir.mkdir(parents=True, exist_ok=True)
    with open(result_dir / f"maxlen_{opts.max_len}_bs_{opts.bs}_lr_{opts.lr}_epoch{opts.epochs}.txt", "w") as fp:
        result = evaluate_model(entities_list, entities_predicted_list)
        print(result)
        json.dump(result, fp)


if __name__ == "__main__":
    main()
