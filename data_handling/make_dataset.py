import re
import math
import torch
import numpy as np

from torch.utils.data import TensorDataset, Subset
from sklearn.model_selection import train_test_split


## Helper functions to process data from a crosstab and split it into training and testing datasets


## The rows of the pandas dataset x represent peptides, and columns represent samples.
def make_dataset_transformer(CrossTab, emb, n_batches, random_state, test_size = 0.20, start_char = "!", padding = "$"):
    ## The "^" character is needed for the amino acids to be extracted from each string properly.
    x = CrossTab
    pattern_dict = "[^a-z*$A-Z0-9!;_-]*"

    ## Helper function to encode position
    def positional_encoding(emb, seq_length, base = 1000):
        encoding = []
        for pos in range(seq_length):
            pos_enc = []
            N = int(emb/2)
            for i in range(N):
                pos_enc = np.append(pos_enc, [math.cos(pos / (base ** (2*i/emb))),
                                              math.sin(pos / (base ** (2*i/emb)))])
            encoding.append(pos_enc)

        encoding = torch.tensor(np.array(encoding))
        return(encoding)


    ## Helper function for splitting feature names into characters
    def split_helper(feature_name):
        return list(filter(None, re.split(pattern_dict, feature_name)))

        
    ## Helper function for masking the appended padding tokens '$'.
    def mask_helper(seq_length, max_pep_length):
        mask = []
        for i in range(max_pep_length):
            new_row = [float(0)] * seq_length + [float('inf')] * (max_pep_length - seq_length)
            mask.append(new_row)
        return(mask)
            
    x.columns = x.columns.astype(str)
    x = x.dropna() # TODO: warning message here

    feature_names_og = x.index
    max_pep_length = max(feature_names_og.map(len))
    ## Adding padding to make all peptide sequences the same length
    feature_names = [start_char + feature_name + padding*(max_pep_length - len(feature_name)) 
                                                        for feature_name in feature_names_og]
    ## Have to refresh the value, as we put a start_char above
    max_pep_length += 1

    masks_peptide = []
    masks_data = []
    ## Have to add 1 because we placed a special token "!" to the beginning of each peptide
    for feature_name in feature_names_og:
      masks_peptide.append(mask_helper(len(feature_name) + 1, max_pep_length))
      masks_data.append(mask_helper(n_batches, n_batches))
    masks_peptide = torch.tensor(masks_peptide)
    masks_data = torch.tensor(masks_data)
    sample_names  = x.columns
    n_features   = len(x) 

    symbols = ''

    for i in range(n_features): 
      symbols = symbols + feature_names[i]

    symbols = ''.join(set(symbols))
    symbols = list(filter(None, re.split(pattern_dict, symbols)))
    symbols_dict = {}

    j = 0
    ## This dictionary will translate the symbols to integers.
    for xx in symbols:
      symbols_dict[xx] = j
      j += 1

    n_letters = len(symbols_dict)

    feature_names = list(map(split_helper, feature_names))
    feature_names = [[symbols_dict[key] for key in feature_name] for feature_name in feature_names]   
    feature_names = torch.tensor(feature_names) 
    positions = positional_encoding(emb, max_pep_length)
    positions = positions.repeat(n_features, 1, 1).float()

    embedding = torch.nn.Embedding(n_letters, emb)
    embedded = embedding(feature_names) + positions

    x = torch.tensor(x.values)
    dataset = TensorDataset(embedded, masks_peptide, x, masks_data) # convert to tensor
    train_idx, test_idx = train_test_split(range(n_features), # make indices
                                          test_size = test_size,
                                          random_state = random_state)

    train_dataset = Subset(dataset, train_idx) # generate subset based on indices
    test_dataset  = Subset(dataset, test_idx)

    metadata = {
      'feature_names_og' : feature_names_og,
      'feature_names'    : feature_names,
      'positions'        : positions,
      'sample_names'     : sample_names,
      'n_features'       : n_features,
      'max_pep_len'      : max_pep_length, 
      'train_idx'        : train_idx,
      'test_idx'         : test_idx,
      'symbols_dict'     : symbols_dict,
      'symbol_embedding' : embedding
      }
      
    return train_dataset, test_dataset, dataset, metadata






