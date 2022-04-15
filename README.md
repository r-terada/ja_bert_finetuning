# 実験結果

## ハイパーパラメータ

|task|model|batch_size|initial_learning_rate|max_epoch|max_length|
|----|-----|----------|---------------------|---------|----------|
|jrte-sentiment|electra|32|1e-4|10|64|
|jrte-sentiment|other|32|5e-5|10|64|
|jrte-entailment|electra|32|1e-4|10|64|
|jrte-entailment|other|32|5e-5|10|64|
|stockmark-ner|electra|32|1e-4|20|256|
|stockmark-ner|other|32|5e-5|20|256|

## 数値評価


### jrte-sentimenft

script: sentiment.py

|model|macro-f1|micro-f1|acc.|test_samples_per_second|
|-----|--------|--------|----|-----------------------|
|cl-tohoku/bert-base-japanese-v2|0.8186|0.8590|0.8590|
|cl-tohoku/bert-base-japanese-whole-word-masking|0.8179|0.8517|0.8517|
|cl-tohoku/bert-large-japanese|0.8032|0.8481|0.8481|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator|0.8392|0.8752|0.8752|
|nlp-waseda/roberta-base-japanese|0.8432|0.8770|0.8770|
|izumi-lab/electra-base-japanese-discriminator|0.8097|0.8499|0.8499|

### jrte-entailment

script: entailment.py

|model|macro-f1|micro-f1|acc.|test_samples_per_second|
|-----|--------|--------|----|-----------------------|
|cl-tohoku/bert-base-japanese-v2|0.8919|0.9005|0.9005|
|cl-tohoku/bert-base-japanese-whole-word-masking|0.8815|0.8915|0.8915|
|cl-tohoku/bert-large-japanese|0.8747|0.8825|0.8825|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator|0.8970|0.9042|0.9042|
|nlp-waseda/roberta-base-japanese|0.8915|0.9005|0.9005|
|izumi-lab/electra-base-japanese-discriminator|0.8885|0.8969|0.8969|

### stockmark-ner

script: train_bio.py

|model|overall-f1|acc.|test_samples_per_second|
|-----|--|----|-----------------------|
|cl-tohoku/bert-base-japanese-v2|0.6179|0.9286|
|cl-tohoku/bert-base-japanese-whole-word-masking|0.6142|0.9234|
|cl-tohoku/bert-large-japanese|0.6102|0.9245|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator|0.5070|0.9048|
|nlp-waseda/roberta-base-japanese|0.7290|0.9474|
|izumi-lab/electra-base-japanese-discriminator|0.5484|0.8969|

※ it should be higher than ~0.8 with f1, there may be bugs in my training script...
cf.) using train_bio_pl.py, which is derived from https://github.com/stockmarkteam/bert-book/blob/master/Chapter8.ipynb, tohoku-large-bert accomplish f1=0.86
