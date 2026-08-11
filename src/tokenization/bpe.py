# %%

from collections import defaultdict
from ..schemas import BPEOutput

def initialize_vocab(corpus: list[str]):
    unique_chars = set()
    for doc in corpus:
        for char in doc:
            unique_chars.add(char)

    vocab = list(unique_chars)
    vocab.sort()
    eow_token = '</n>'
    # Indicating end of a word
    vocab.append(eow_token)
    return vocab



word_freqs = defaultdict(int) # if specified key is not found instead of throwing an error constructs it.

def count_word_freqs(corpus):

    for doc in corpus:

        words = doc.split(' ')

        for word in words:

            word_freqs[word] += 1


    return word_freqs 


# %%

word_splits = {}

def create_word_splits(word_freqs: dict[str, int]):

    for word, _ in word_freqs.items():

        word_splits[word] = list(word)
    return word_splits



# %%

def count_merge_pairs(word_splits: dict[str, list[str]]):
    merge_pairs = defaultdict(int)

    for word, chars in word_splits.items():

        freq = word_freqs[word]

        for i in range(len(chars)-1):

            pair = (chars[i], chars[i+1])

            merge_pairs[pair] +=freq
    return merge_pairs


# %%


def apply_merge(word_splits: dict[str, list[str]], pair: tuple[str, str]):

    char1, char2 = pair

    merged = char1 + char2
    

    for _, chars in word_splits.items():

        i = 0

        new_chars = []

        while i < len(chars):

            if i < len(chars)-1 and char1 == chars[i] and char2 == chars[i+1]:

                new_chars.append(merged)

                i+=2

            else:

                new_chars.append(chars[i])

                i+=1

        chars[:] = new_chars

    return word_splits


# %% 

merge_rules = []


def bpe_training(corpus: list[str], num_of_merges: int) -> BPEOutput:
    vocab = initialize_vocab(corpus)

    word_freqs = count_word_freqs(corpus)

    word_splits = create_word_splits(word_freqs)

    for _ in range(num_of_merges):
        merge_pairs = count_merge_pairs(word_splits)

        best_pair = max(merge_pairs, key=merge_pairs.get)

        merge_rules.append(best_pair)

        apply_merge(word_splits, best_pair)    

    vocab.extend(merge_rules)
    token_to_id = {token: idx for idx, token in enumerate(vocab)}
    
    return word_splits, vocab, token_to_id, merge_rules
    




def bpe_inference(corpus, merge_rules,token_to_id):

    all_tokens = []

    all_ids = []


    for doc in corpus:

        words = doc.split(' ')

        doc_tokens = []


        for word in words:
            chars = list(word)


            for pair in merge_rules:

                splits = apply_merge({word: chars}, pair)[word]


            doc_tokens.extend(splits)


        doc_ids = [token_to_id[token] for token in doc_tokens if token in token_to_id]


        all_tokens.append(doc_tokens)

        all_ids.append(doc_ids)


    return all_tokens, all_ids
    

"""
print(word_splits)
print(len(merge_rules))
print(merge_rules)

"""
