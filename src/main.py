from .tokenization.bpe import bpe_training, bpe_inference
from .model.transformer import TransformerModel
import torch
with open("data/yunusemre.txt","r",encoding="utf-8") as file:
    corpus = file.read()

corpus=[corpus]
word_splits, vocab, token_to_id, merge_rules = bpe_training(corpus, 30)
vocab_size = len(vocab)

tokens, all_ids = bpe_inference(corpus, merge_rules, token_to_id)

print(len(all_ids),len(all_ids[0]))


# Önce listeyi numpy dizisine çevirin
all_ids_np = torch.tensor(all_ids)

# Artık iki boyutlu dilimleme yapabilirsiniz
idx = all_ids_np[:, :512]
target = all_ids_np[:, 1:513]

idx= idx.to("cuda")
target= target.to("cuda")

model = TransformerModel(vocab_size=vocab_size).to("cuda")

logits, loss = model(idx,target)

print(loss)