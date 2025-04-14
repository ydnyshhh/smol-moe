import torch
from typing import Tuple
from src.data.tokenizer import Tokenizer

# Path to the raw text file for training
DATA_FILE = "dataset/tiny_stories_train.txt"

class Loader:
    """
    Loads raw text, encodes it using a tokenizer, splits into train/val sets,
    and provides methods to generate batches for training.
    """

    def __init__(self, tokenizer: Tokenizer, train_split_size: float = 0.9) -> None:
        # Load the entire text file into memory
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            text = f.read()

        # Tokenize the text and convert to tensor of token IDs
        data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

        # Split into train and validation sets
        train_data_size = int(train_split_size * len(data))
        self.train_data = data[:train_data_size]
        self.val_data = data[train_data_size:]

    def get_batch(
        self, split: str, batch_size: int, block_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a batch of input (x) and target (y) sequences.

        Args:
            split: 'train' or 'val'
            batch_size: number of sequences in a batch
            block_size: length of each input sequence

        Returns:
            x: input tokens of shape (batch_size, block_size)
            y: target tokens (next-token labels) of shape (batch_size, block_size)
        """
        data = self.train_data if split == "train" else self.val_data

        # Random starting indices for each sequence
        ix = torch.randint(len(data) - block_size, (batch_size,))

        # Create input and target sequences
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])  # next-token

        return x, y
