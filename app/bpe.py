corpus = [
    "Artificial intelligence is transforming the world.",
    "Cats often nap in sunny windows during the afternoon.",
    "Quantum mechanics challenges our classical understanding of reality.",
    "A warm cup of coffee can brighten even the coldest morning.",
    "Exploring distant planets demands innovative engineering solutions."
]

# %%
unique_chars = set()

for doc in corpus:
    for char in doc:
        unique_chars.add(char)


vocab = list(unique_chars)
vocab.sort()

# Indicating end of a word
vocab.append('<eow>')




# %%

word_splits = {}

for doc in corpus:
    for word in doc.split():
        











