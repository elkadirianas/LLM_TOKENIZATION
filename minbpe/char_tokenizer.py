from .base import Tokenizer

class CharTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text, vocab_size=None, verbose=False):
        unique_chars = sorted(set(text))
        self.vocab = {i: ch.encode('utf-8') for i, ch in enumerate(unique_chars)}
        self.char_to_id = {ch: i for i, ch in enumerate(unique_chars)}

    def encode(self, text):
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids):
        return ''.join([self.vocab[i].decode('utf-8') for i in ids])

