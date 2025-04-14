class Tokenizer:
    """
    A simple character-level tokenizer that maps each character
    to a unique integer ID and vice versa.
    """

    def __init__(self) -> None:
        # Define the character vocabulary (can be extended)
        self.char_string = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

        # Remove duplicates and sort
        self.chars = sorted(list(set(self.char_string)))

        # Vocabulary size
        self.vocab_size = len(self.chars)

        # Character to index mapping
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # string → index
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # index → string

        # Encoding and decoding functions
        self.encode = lambda string: [self.stoi[c] for c in string]  # text → list of IDs
        self.decode = lambda tokens: "".join([self.itos[i] for i in tokens])  # IDs → text
