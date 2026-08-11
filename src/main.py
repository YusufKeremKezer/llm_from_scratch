from .tokenization.bpe import bpe_training, bpe_inference


with open("data/yunusemre.txt","r",encoding="utf-8") as file:
    corpus = file.read()

corpus = [corpus]

word_splits, vocab, token_to_id, merge_rules = bpe_training(corpus, 30)
vocab_size = len(vocab)

tokens, all_ids = bpe_inference(corpus, merge_rules, token_to_id)

print(len(all_ids),len(all_ids[0]))