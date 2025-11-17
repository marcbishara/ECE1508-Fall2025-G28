import re
from collections import Counter

def repetition_penalty(text, max_repetition_ratio=0.2):
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    repeated = 0
    for count in counts.values():
        if count > 1:
            repeated += count - 1
    repetition_ratio = repeated / len(tokens)
    return -max(0, repetition_ratio - max_repetition_ratio)

def length_penalty(text, min_len=5, max_len=25):
    tokens = re.findall(r"\w+", text)
    length = len(tokens)
    if length == 0:
        return -1.0
    if length < min_len:
        return -((min_len - length) / min_len)
    if length > max_len:
        return -((length - max_len) / max_len)
    return 0.0
    