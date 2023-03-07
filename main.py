import string
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, file_path, seq_length=50):
        self.seq_length = seq_length
        with open(file_path, 'r', encoding='utf8') as f:
            text = f.read()
        # Remove punctuation and convert to lowercase
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        # Create a dictionary to map words to integers
        self.word_to_index = {word: i for i, word in enumerate(sorted(set(text.split())))}
        # Create a list of integers representing the text
        self.data = [self.word_to_index[word] for word in text.split()]

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data[index:index+self.seq_length]
        y = self.data[index+self.seq_length]
        return np.array(x), np.array(y)
