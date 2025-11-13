# RL-based Finetuning of Language Models

Reinforcement Learning with Human Feedback (RLHF) is implemented by Proximal Policy Optimization (PPO) to improve language model outputs according to a reward model.

## Setup
pip install torch
pip install transformers

## Data Sources
Sarcasm datasets:

Sarcasm_Headlines_Dataset.json
Sarcasm_Headlines_Dataset_v2.json
https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection

test-balanced.csv
test-unbalanced.csv
train-balanced-sarasm.csv
https://www.kaggle.com/datasets/danofer/sarcasm

## Reward Model
[todo]

## Project Scripts
[todo]

## Literature Review
[Prior art search on fine tuning LM with RLHF and PPO](prior_art_research.md)


## TODO

### Reward model

#### Sarcasm model [Tamara]
- identify/try different datasets
- any necessary data cleaning/engineering
- train/fine-tune classifier for detecting sarcasm

#### Subjective/objective model [Charles]
- identify/try different datasets
- any necessary data cleaning/engineering
- train/fine-tune classifier for detecting objective/subjective

#### Other reward signals
- repeating words penalty
- length of sentence penalty

### PPO
- overall architecture:
  - policy model
  - reference model
  - reward model (from above)
  - value model
  - GAE
 
### Baseline SFT
- model for supervised finetuning to use as baseline

### Quantitative comparison of RL and SFT
- compare using suitable metrics
- plots, etc

### Qualitative comparison of RL and SFT
- produce some examples for human qualitative comparison

### Ablation experiments
- figure out what aspects to remove for ablation -- maybe certain of the reward signals?

### Final report
- 5 page final report

### Final presentation
- 5 min final presentation
