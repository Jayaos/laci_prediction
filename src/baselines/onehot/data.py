import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LacIDatasetOneHot(Dataset):
    """
    LacI dataset for one-hot model
    """
    
    def __init__(self, data_dict, aa2id, id2aa):
        self.aa2id = aa2id
        self.id2aa = id2aa
        self.data_dict = data_dict
        self.input_keys = list(data_dict.keys())

    def __len__(self):
        return len(self.input_keys)

    def __getitem__(self,idx):
        
        sequence = self.input_keys[idx]
        mutation, label = self.data_dict[sequence]
        onehot_encoded_sequence = np.array(np.eye(len(self.aa2id))[[self.aa2id[aa] for aa in sequence]]).reshape(-1)

        return (mutation, self.input_keys[idx]), onehot_encoded_sequence, label 

def collate_function_onehot(batch):
    
    mutation_sequence_pair_batch, onehot_encoded_sequence_batch, label_batch = list(zip(*batch))
    
    return mutation_sequence_pair_batch, torch.from_numpy(np.array(onehot_encoded_sequence_batch)).float(), torch.from_numpy(np.array(label_batch)).float()
