# %%

from collections import defaultdict
corpus = [
    "Artificial intelligence is transforming the world.",
    "Cats often nap in sunny windows during the afternoon.",
    "Quantum mechanics challenges our classical understanding of reality.",
    "A warm cup of coffee can brighten even the coldest morning.",
    "Exploring distant planets demands innovative engineering solutions."
]

unique_chars = set()

for doc in corpus:
    for char in doc:
        unique_chars.add(char)


vocab = list(unique_chars)
vocab.sort()


eow_token = '</n>'
# Indicating end of a word
vocab.append(eow_token)

print(len(vocab))


# %%
word_freqs = defaultdict(int) # if specified key is not found instead of throwing an error constructs it.
def count_word_freqs(corpus):
    for doc in corpus:
        words = doc.split(' ')
        for word in words:
            word_freqs[word] += 1

    return word_freqs 

# %%
word_splits = {}
def create_word_splits(word_freqs):
    for word, _ in word_freqs.items():
        word_splits[word] = list(word)
    return word_splits


# %%
merge_pairs = defaultdict(int)
def count_merge_pairs(word_splits):
    for word, chars in word_splits.items():
        freq = word_freqs[word]
        for i in range(len(chars)-1):
            pair = (chars[i], chars[i+1])
            merge_pairs[pair] +=freq
    return merge_pairs

# %%

def apply_merge(word_splits, pair):
    char1, char2 = pair
    merged = char1 + char2
    for _, chars in word_splits.items():
        for i in range(len(chars)-1):
            if chars[i] == char1 and chars[i+1] == char2:
                chars[i] = merged
                chars.pop(i+1)
                break
    return word_splits

# %% 
merge_rules = []
def bpe(corpus, num_of_merges):
    for i in range(num_of_merges):
        word_freqs = count_word_freqs(corpus)
        word_splits = create_word_splits(word_freqs)
        merge_pairs = count_merge_pairs(word_splits)
        best_pair = max(merge_pairs, key=merge_pairs.get)
        merge_rules.append(best_pair)
        apply_merge(word_splits, best_pair)    
    return word_splits
    
word_splits = bpe(corpus, 30)
# %%

# %%
