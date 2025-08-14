# %%
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




# %%
# Counting the number of times each character appears in the corpus
def count_pair(corpus):
    word_splits = {}

    for doc in corpus:
        words = doc.split(' ')
        for word in words:
            char_list = list(word) + [eow_token]
            char_list = tuple(char_list) # tuple for immutability otherwise its not accepted by dictionary
            word_splits[char_list] = word_splits.get(char_list, 0) + 1
    
    return word_splits

word_splits = count_pair(corpus)

    

# %%
def merge_pair(word_splits):
    pair_counts = {}
    for word_split, count in word_splits.items():
        for i in range(len(word_split) - 1):
            pair = (word_split[i], word_split[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
    
    return pair_counts



paired_chars = merge_pair(word_splits)



# %%
print(paired_chars)
# %%
merge_limit = 3
to_be_merged_list = [key for key, value in paired_chars.items() if value >= merge_limit]

print(to_be_merged_list)
# %% 
def pair_to_merge(merge_list,word_splits):
    
    for pair in merge_list:
        str1,str2 = pair
        merged_pair = str1 + str2
        print(merged_pair)


merged_list = pair_to_merge(to_be_merged_list)

print(list(word_splits.keys())[0][:4])







# %%
