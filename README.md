# Generally Sarcastic Transformer: Using RL to Fine-tune a Language Model for Sarcasm

## Overview
This project presents the Generally Sarcastic Transformer (GST), a GPT-2-based language model first fine-tuned via supervised learning and then further optimized with Proximal Policy Optimization (PPO) to respond sarcastically to given prompts. We design a multi-signal reward model that combines:
- Sarcasm Classifier Score
- Subjectivity Classifier Score
- Repetition Words Score
- Length of Sentence Score

The model is trained on the Sarcasm on Reddit dataset. Experimental results show that the proposed method effectively improves the model's reward performance, demonstrating the applicability of RL for stylistic text generation. 

## Project Architecture
This project uses the PPO algorithm provided by the `trl` library. The training loop optimizes the policy based on a weighted sum of four distinct reward signals.

### Models
- Policy Model (Language Model): Supervised fine-tuned GPT-2 model
  - Training code: Located in the [`sft_baseline/`](./sft_baseline) directory
  - Model weights: Hosted on Hugging Face at [`Zoe3324/gpt2-sft-full`](https://huggingface.co/Zoe3324/gpt2-sft-full)
- Reference Model: A frozen copy of the Policy Model (used to calculate KL-divergence)
- Sarcasm Classifier: DistilBERT-based classifier trained to detect sarcasm
  - Training code: Located in the [`sarcasm_classifier/`](./sarcasm_classifier) directory
  - Model weights: Hosted on Hugging Face at [`tmrcnl/SarcasmRewardModel`](https://huggingface.co/tmrcnl/SarcasmRewardModel)
- Subjectivity Classifier: A custom CNN classifier using GloVe embeddings to detect subjective/objective language
  - [TODO]

### The Reward Model
The total reward $R_{total}$ for a given response is calculated as:

$$Total Reward Score = \text(Sarcasm Score + 0.5 \cdot Subjectivity Classifier Score - Repetition Score + 0.1 \cdot Length Score)$$

Where:
1.  **$Sarcasm Score$**: Probability score from the Sarcasm Classifier.
2.  **$Subjectivity Score$**: Probability score from the Subjectivity Classifier CNN (objectivity score inverted to reward subjectivity).
3.  **$Repetition Score$**: A penalty score calculated based on token n-gram repetition.
4.  **$Length Score$**: A penalty score if the response is too short or too long.

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
To set up this project:
### 1. Clone the repository
```bash
git clone https://github.com/marcbishara/ECE1508-Fall2025-G28.git
cd ECE1508-Fall2025-G28
```
### 2. Install required packages
```bash
pip install -r requirements.txt
```
Important Note: This project specifically requires trl==0.11.4 or earlier due to changes in newer versions.

### 3. Environment Variable
Set up your Hugging Face and Weights and Biases tokens to load models and track run metrics.

## How to use
The main PPO training pipeline is implemented in the script which contains the full workflow for training the GST using PPO: 
- `PPO_TRLwDL_GST.ipynb`

To run the training workflow, execute the notebook.

Training Configuration:
- Optimizer: PPO (Proximal Policy Optimization)
- Learning Rate: 1.41e-5 (Adjustable in config)
- Batch Size: 512
- Mini-batch Size: 64
- Epochs: 2
- Tracker: Weights & Biases (wandb.ai)

## Data Sources

The project uses the Sarcasm on Reddit dataset (https://www.kaggle.com/datasets/danofer/sarcasm), formatted for PPO training.
- Hugging Face ID: `marcbishara/sarcasm-on-reddit`
- Split for PPO training: `ppo_train`
- Other splits used to train Sarcasm Classifier, fine-tune GPT-2, and as holdout

## References

- Literature review: [Prior art search on fine tuning LM with RLHF and PPO](prior_art_research.md)
- TRL Library: https://huggingface.co/docs/trl/v0.11.4/en/index
