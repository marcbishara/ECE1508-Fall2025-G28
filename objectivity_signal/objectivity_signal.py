"""
Call function objectivity_reward(sentence) for an objectivity reward score.
The input is a string. e.g.
sentence = "I feel happy"

There is a one time setup required. Run download_models() to download the model parameters.
The files are too big to be uploaded to GitHub.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from tqdm import tqdm
import numpy as np
import os
import gdown

# Objectivity Classifier
class objectivity_classifier(torch.nn.Module):
    def __init__(self, embeddings, k1, k2, n1, n2):
        super().__init__()

        embedding_dim = len(embeddings[0])
        self.embeddings = nn.Embedding.from_pretrained(embeddings, freeze=True)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n1, kernel_size=(k1, embedding_dim), bias=False)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=n2, kernel_size=(k2, embedding_dim), bias=False)
        self.fc = nn.Linear(n1 + n2, 1)

    def forward(self, x):
        embeddings = self.embeddings(x).unsqueeze(1) # (batch, 1, num_words, em_dim)
        # CNN - parameter: (batch, channel, height, width)
        k1_out = F.relu(self.conv1(embeddings)) # (batch, n1, L, 1)
        k2_out = F.relu(self.conv2(embeddings)) # (batch, n2, L, 1)
        # Max pooling
        k1_out = F.max_pool2d(k1_out, (k1_out.shape[2], 1)) # (batch, n1, 1, 1)
        k2_out = F.max_pool2d(k2_out, (k2_out.shape[2], 1)) # (batch, n2, 1, 1)
        # Organize
        k1_out = k1_out.squeeze(3).squeeze(2) # (batch, n1)
        k2_out = k2_out.squeeze(3).squeeze(2) # (batch, n2)
        # fc
        out = torch.cat([k1_out, k2_out], dim=1)
        out = self.fc(out)

        return out

def load_glove_vectors(glove_path, vocab_size=None):
    """
    Load GloVe vectors from file into PyTorch tensors

    Args:
        glove_path: Path to GloVe text file
        vocab_size: Number of words to load (None for all)

    Returns:
        word2idx: Dictionary mapping words to indices
        idx2word: List mapping indices to words
        embeddings: PyTorch tensor of embeddings
    """
    print(f"Loading GloVe vectors from {glove_path}...")

    word2idx = {}
    idx2word = []
    vectors = []

    with open(glove_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
            if vocab_size and i >= vocab_size:
                break

            values = line.strip().split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')

            word2idx[word] = i
            idx2word.append(word)
            vectors.append(vector)

    embeddings = torch.from_numpy(np.array(vectors))

    print(f"Loaded {len(word2idx)} words with dimension {embeddings.shape[1]}")

    return word2idx, idx2word, embeddings

# This function is only to be run once for the trianed model parameters.
def download_models():
    MODEL_DIR = "objectivity_signal"
    os.makedirs(MODEL_DIR, exist_ok=True)

    GLOVE_FILE_ID = "1ufQLwedjFzjmRej-Qfp0MyM2iOH6oP9U"
    MODEL_FILE_ID = "1EGvEGgZwJVJLBWjQcGWCfYV-kp9QrZcv"

    MODEL_PATH = os.path.join(MODEL_DIR, "model_CNN_objectivity_classifier.pt")
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}"
        print("Downloading model...")
        gdown.download(url, MODEL_PATH, quiet=False)
    else:
        print("Model already exists!")

    GLOVE_PATH = os.path.join(MODEL_DIR, "glove.6B.100d.txt")
    if not os.path.exists(GLOVE_PATH):
        url = f"https://drive.google.com/uc?id={GLOVE_FILE_ID}"
        print("Downloading GloVe...")
        gdown.download(url, GLOVE_PATH, quiet=False)
    else:
        print("GloVe already exists!")

def objectivity_reward(sentence):
    # Be careful of file location change or name change.
    embeddings_path = "./objectivity_signal/glove.6B.100d.txt"
    model_path = "./objectivity_signal/model_CNN_objectivity_classifier.pt"

    word2idx, idx2word, embeddings = load_glove_vectors(embeddings_path)

    model_CNN = objectivity_classifier(embeddings, k1=2, k2=4, n1=100, n2=100)
    model_CNN.load_state_dict(torch.load(model_path))

    sentence = re.sub(r'[^\w\s]', '', sentence)
    V = len(word2idx)

    tokens = torch.tensor(
        [word2idx.get(word, V-1) for word in sentence.lower().split()] + [0]*4,
        dtype=torch.long
    ).unsqueeze(0)

    prob = torch.sigmoid(model_CNN(tokens)).squeeze(0).squeeze(0) # This is a Tensor. e.g. tensor(0.9336, grad_fn=<SqueezeBackward1>)
    
    reward = round(prob.item(), 4) # Keep 4 decimal places

    return reward


# Run this file to see an example output. 
# Everything above is just function definition.
# Functions can be used in other files. Most likely, objectivity_reward(sentence) is the only function we need to call for this reward signal.
sentence = "I feel happy"
prob = objectivity_reward(sentence)
print(prob)

