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
|cl-tohoku/bert-base-japanese-v2|0.8186|0.8590|0.8590|101.7830|
|cl-tohoku/bert-base-japanese-whole-word-masking|0.8179|0.8517|0.8517|100.6520|
|cl-tohoku/bert-large-japanese|0.8032|0.8481|0.8481|37.1100|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator|0.8392|0.8752|0.8752|98.9070|
|nlp-waseda/roberta-base-japanese|0.8432|0.8770|0.8770|99.3320|
|izumi-lab/electra-base-japanese-discriminator|0.8097|0.8499|0.8499|99.4560|

### jrte-entailment

script: entailment.py

|model|macro-f1|micro-f1|acc.|test_samples_per_second|
|-----|--------|--------|----|-----------------------|
|cl-tohoku/bert-base-japanese-v2|0.8919|0.9005|0.9005|101.1660|
|cl-tohoku/bert-base-japanese-whole-word-masking|0.8815|0.8915|0.8915|101.8050|
|cl-tohoku/bert-large-japanese|0.8747|0.8825|0.8825|37.1160|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator|0.8970|0.9042|0.9042|99.2990|
|nlp-waseda/roberta-base-japanese|0.8915|0.9005|0.9005|100.3370|
|izumi-lab/electra-base-japanese-discriminator|0.8885|0.8969|0.8969|98.8900|

### max-length tuning on sentiment

|model|macro-f1|micro-f1|acc.|test_samples_per_second|
|-----|--------|--------|----|-----------------------|
|waseda-maxlen64|0.8432|0.8770|0.8770|99.3320|
|waseda-maxlen128|0.8496|0.8770|0.8770|81.7610|
|megagon-maxlen64|0.8392|0.8752|0.8752|98.9070|
|megagon-maxlen128|0.8594|0.8897|0.8897|80.4180|

