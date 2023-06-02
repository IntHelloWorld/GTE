import pickle

vocab_dir = "/home/qyh/projects/GTE/vocabulary/token_to_index_Clone.pkl"
token_to_idx = pickle.load(open(vocab_dir, "rb"))
vocab_size = len(token_to_idx)
for k in token_to_idx.keys():
    if "expression" in k:
        print(k)
print(vocab_size)
