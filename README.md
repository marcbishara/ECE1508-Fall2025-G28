# Generally Sarcastic Transformer: Using RL to Fine-tune a Language Model for Sarcasm

## Overview
This project presents the Generally Sarcastic Transformer (GST), a GPT-2-based language model first fine-tuned via supervised learning and then further optimized with Proximal Policy Optimization (PPO) to respond sarcastically to given prompts. We design a multi-signal reward model that combines:
- Sarcasm Classifier Score
- Subjectivity Classifier Score
- Repetition Words Score
- Length of Sentence Score

The model is trained on the Sarcasm on Reddit dataset. Experimental results show that the proposed method effectively improves the model's reward performance, demonstrating the applicability of RL for stylistic text generation. 

## Project Structure
The repository is organized as follows:
```text
.
├── Archive/                       # Archived early-stage experiments and unused scripts
├── Screenshots of good samples/   # Example outputs
├── evaluation/                    # Scripts for evaluating model performance (including avg reward score, lexical richness, and diversity)
├── objectivity_signal/            # Subjectivity classifier for reward model
├── sarcasm_classifier/            # Sarcasm classifier for reward model
├── sft_baseline/                  # Supervised fine-tuning (SFT) baseline models
├── Dataset_prep_script.ipynb      # Dataset preprocessing and cleaning
├── PPO_TRLwDL_GST.ipynb           # Main PPO training pipeline for GST
├── repetition_length_signal.py    # Repetition and length penalty reward functions
├── prior_art_research.md          # Literature review and prior work summary
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore configuration
```

## Setup and installation
To set up for this project:
### 1. Clone the repository
```bash
git clone https://github.com/marcbishara/ECE1508-Fall2025-G28.git
cd ECE1508-Fall2025-G28
```
### 2. Install required packages
```bash
pip install -r requirements.txt
```

## How to use
The main PPO training pipeline is implemented in the script which contains the full workflow for training the GST using PPO: 
- `PPO_TRLwDL_GST.ipynb`


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
