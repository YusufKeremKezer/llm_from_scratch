from .model.transformer import TransformerModel
from .tokenization.bpe import bpe_training, bpe_inference, BPEResult


with open(data/yunusemre.txt,"r",encoding="utf-8") as file:
    corpus = file.read()


word_splits, vocab, token_to_id = bpe_training(corpus, 30)
vocab_size = len(vocab)

model = TransformerModel(vocab_size=vocab_size)

model(idx, targets)

def main():
    model()

if __name__ == "__main__":
    main()
