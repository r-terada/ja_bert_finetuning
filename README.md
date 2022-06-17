# How to use

## Install

- `docker compose build`
- `docker compose up`
- `git clone https://github.com/megagonlabs/jrte-corpus`

## Run

- `python jrte/sentiment.py --model-name nlp-waseda/roberta-base-japanese`


# 実験結果

## ハイパーパラメータ


### sentiment

|model|batch_size|initial_learning_rate|gradient_accumulation_steps|max_length|
|-----|----------|---------------------|---------------------------|----------|
|bert-base-japanese-v2|32|5e-05|1|128|
|bert-base-japanese-v2(512)|32|5e-05|1|512|
|bert-base-japanese-whole-word-masking|32|5e-05|1|128|
|bert-large-japanese|32|5e-05|1|128|
|bert-base-japanese-char-v2(512)|16|7e-05|1|512|
|transformers-ud-japanese-electra-base-discriminator|32|0.0001|1|128|
|transformers-ud-japanese-electra-base-discriminator(512)|32|0.0001|1|512|
|roberta-base-japanese|32|5e-05|1|128|
|roberta-base-japanese(512)|32|5e-05|1|512|
|roberta-large-japanese|16|2e-05|1|128|
|roberta-large-japanese(512)|4|1e-05|1|512|
|electra-base-japanese-discriminator|32|0.0001|1|128|
|chiTra-1.0|32|5e-05|1|128|

### reputation

|model|batch_size|initial_learning_rate|gradient_accumulation_steps|max_length|
|-----|----------|---------------------|---------------------------|----------|
|bert-base-japanese-v2|32|5e-05|1|128|
|bert-base-japanese-v2(512)|32|5e-05|1|512|
|bert-base-japanese-whole-word-masking|32|5e-05|1|128|
|bert-large-japanese|32|5e-05|1|128|
|bert-large-japanese(512, batch_size=16)|16|5e-05|1|512|
|bert-base-japanese-char-v2(512)|16|7e-05|1|512|
|transformers-ud-japanese-electra-base-discriminator|32|0.0001|1|128|
|transformers-ud-japanese-electra-base-discriminator(512)|32|0.0001|1|512|
|roberta-base-japanese|32|5e-05|1|128|
|roberta-base-japanese(512)|32|5e-05|1|512|
|roberta-large-japanese|16|2e-05|1|128|
|roberta-large-japanese(512)|4|1e-05|1|512|
|electra-base-japanese-discriminator|32|0.0001|1|128|
|chiTra-1.0|32|5e-05|1|128|
|chiTra-1.0(512)|32|5e-05|1|512|

### entailment

|model|batch_size|initial_learning_rate|gradient_accumulation_steps|max_length|
|-----|----------|---------------------|---------------------------|----------|
|bert-base-japanese-v2|32|5e-05|1|128|
|bert-large-japanese|16|2e-05|1|128|
|bert-base-japanese-char-v2(512)|16|7e-05|1|512|
|transformers-ud-japanese-electra-base-discriminator|32|0.0001|1|128|
|transformers-ud-japanese-electra-base-discriminator(512)|16|0.0001|2|512|
|roberta-base-japanese|32|1e-05|1|128|
|roberta-base-japanese(512)|16|1e-05|2|512|
|roberta-large-japanese|16|7e-06|1|128|
|roberta-large-japanese(512)|4|5e-06|1|512|
|chiTra-1.0|32|5e-05|1|128|

## 数値評価


### jrte-sentimenft

script: jrte/sentiment.py

|model(max_len)|macro-f1|weighted-f1|acc.|
|-----|--------|--------|----|
|cl-tohoku/bert-base-japanese-v2|0.8173|0.8559|0.8590|
|cl-tohoku/bert-base-japanese-v2(512)|0.8438|0.8697|0.8716|
|cl-tohoku/bert-base-japanese-whole-word-masking|0.8150|0.8444|0.8499|
|cl-tohoku/bert-large-japanese|0.8026|0.8424|0.8463|
|cl-tohoku/bert-base-japanese-char-v2(512)|0.8106|0.8543|0.8571|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator|**0.8594**|**0.8889**|**0.8897**|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator(512)|0.8530|0.8793|0.8807|
|nlp-waseda/roberta-base-japanese|0.8496|0.8719|0.8770|
|nlp-waseda/roberta-base-japanese(512)|0.8576|0.8832|0.8843|
|nlp-waseda/roberta-large-japanese|**0.8753**|**0.8928**|**0.8951**|
|nlp-waseda/roberta-large-japanese(512)|0.8572|0.8830|0.8843|
|izumi-lab/electra-base-japanese-discriminator|0.8116|0.8422|0.8463|
|chiTra-1|0.8214|0.8594|0.8608|

### jrte-reputation

script: reputation.py

|model(maxlen)|macro-f1|weighted-f1|acc.|
|-----|--------|--------|----|
|cl-tohoku/bert-base-japanese-v2|0.8951|0.9023|0.9024|
|cl-tohoku/bert-base-japanese-v2(512)|0.8959|0.9035|0.9042|
|cl-tohoku/bert-base-japanese-whole-word-masking|0.8938|0.9016|0.9024|
|cl-tohoku/bert-large-japanese|0.9017|0.9089|0.9096|
|cl-tohoku/bert-large-japanese(512, batch_size=16)|0.8846|0.8928|0.8933|
|cl-tohoku/bert-base-japanese-char-v2(512)|0.8816|0.8912|0.8933|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator|0.8915|0.8987|0.8987|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator(512)|0.8936|0.9015|0.9024|
|nlp-waseda/roberta-base-japanese|0.8984|0.9056|0.9060|
|nlp-waseda/roberta-base-japanese(512)|0.8933|0.9013|0.9024|
|nlp-waseda/roberta-large-japanese|0.8918|0.9005|0.9024|
|nlp-waseda/roberta-large-japanese(512)|**0.9042**|**0.9110**|**0.9114**|
|izumi-lab/electra-base-japanese-discriminator|0.8917|0.8997|0.9005|
|chiTra-1|**0.9054**|**0.9124**|**0.9132**|
|chiTra-1(512)|0.9052|0.9123|0.9132|

### jrte-entailment

script: entailment.py

|model(maxlen)|macro-f1|weighted-f1|acc.|
|-----|--------|--------|----|
|cl-tohoku/bert-base-japanese-v2|0.8847|0.8851|0.8850|
|cl-tohoku/bert-large-japanese|0.8879|0.8883|0.8882|
|cl-tohoku/bert-base-japanese-char-v2(512)|0.8813|0.8817|0.8815|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator|0.8967|0.8970|0.8969|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator(512)|0.8965|0.8968|0.8967|
|nlp-waseda/roberta-base-japanese|**0.9078**|**0.9080**|**0.9079**|
|nlp-waseda/roberta-base-japanese(512)|0.9031|0.9034|0.9032|
|nlp-waseda/roberta-large-japanese|0.9031|0.9034|0.9032|
|nlp-waseda/roberta-large-japanese(512)|0.9047|0.9051|0.9050|
|chiTra-1|0.8899|0.8903|0.8902|

