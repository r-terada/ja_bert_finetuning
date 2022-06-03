## Hyperparameters 

|model|dataset|epoch|batch_size|learning_rate|warmup_ratio|max_length|
|-------|-----|-----|----------|-------------|------------|----------|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator|MARC_ja|2|16|3e-5|0.1|512|
||JSTS|4|16|3e-5|0.1|512|
||JNLI|3|16|3e-5|0.1|512|
||JSQuAD|5|16|3e-5|0.1|384|
||JCommonsenseQA|5|16|1e-4|0.1|64|

## Scores 

|model|MARC_ja|JSTS|JNLI|JSQuAD|JCommonsenseQA|
|-----|-------|----|----|------|--------------|
||acc.|Pearson/Spearman|acc.|EM/F1|acc.|
|best score|0.964|0.923/0.891|0.924|0.897/0.947|0.901|
|megagonlabs/transformers-ud-japanese-electra-base-discriminator|0.963|0.913/0.877|0.921|0.795/0.887|0.856|

