import torch
import numpy as np
import copy
from src.utils import *
from tqdm import tqdm
from torch.utils.data import Dataset
from tape import TAPETokenizer


class RepresentatonDataset(Dataset):
    
    def __init__(self, mutation_data_dict, representation_data_dict):

        def _get_target_representation_data_dict(mutation_data_dict, representation_data_dict):

            target_representation_data_dict = dict()

            for mut_seq in list(mutation_data_dict.keys()):
                target_representation_data_dict[mut_seq] = copy.deepcopy(representation_data_dict[mut_seq])
        
            return target_representation_data_dict
    
        self.mutation_data_dict = mutation_data_dict
        self.target_representation_data_dict = _get_target_representation_data_dict(mutation_data_dict, representation_data_dict)
        self.input_keys = list(mutation_data_dict.keys())

    def __len__(self):
        return len(self.input_keys)
    
    def __getitem__(self, idx):

        mutation, label = self.mutation_data_dict[self.input_keys[idx]]
        wt_residue, position, mt_residue = mutation[0], int(mutation[1:-1]), mutation[-1]
        representaton = self.target_representation_data_dict[self.input_keys[idx]]

        # return (mutation, sequence) and bin_label
        return wt_residue, position, mt_residue, self.input_keys[idx], representaton, label


def collate_function_representation_dataset(batch):
    
    wt_residue, position, mt_residue, sequence, representation, label = list(zip(*batch))
    
    return list(wt_residue), torch.from_numpy(np.array(position)).long(), list(mt_residue), list(sequence), torch.from_numpy(np.array(representation)), torch.from_numpy(np.array(label)).float()


def representation_data_dict_to_array(mutation_data_dict, representation_data_dict):
    
    representation_array = []
    label_array = []
    
    for mut_seq, item in mutation_data_dict.items():
        
        representation = representation_data_dict[mut_seq]
        label = item[1]
        
        representation_array.append(representation)
        label_array.append(label)
        
    return (np.array(representation_array), np.array(label_array))


class SequenceDataset(Dataset):
    
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.input_keys = list(data_dict.keys())

    def __len__(self):
        return len(self.input_keys)

    def __getitem__(self, idx):

        mutation, label = self.data_dict[self.input_keys[idx]]

        # return (mutation, sequence) and bin_label
        return (mutation, self.input_keys[idx]), label
    

def collate_function_sequence_dataset(batch):
    
    mutation_sequence_pair, label = list(zip(*batch))
    
    return mutation_sequence_pair, torch.from_numpy(np.array(label))


def batch_convert_unirep(tokenizer: TAPETokenizer, mutation_sequence_pair_batch):
    """
    encode SequenceDataset batched output using TAPETokenizer
    this is required since TAPETokenizer is not compatible to encode the batched output of SequenceDataset
    """

    encoded_sequence_list = []

    for mut, seq in mutation_sequence_pair_batch:
        encoded_sequence_list.append(tokenizer.encode(seq))

    return torch.from_numpy(np.array(encoded_sequence_list)).long()


def batch_convert_unirep_sequence(tokenizer: TAPETokenizer, mutation_sequence_batch):
    """
    encode SequenceDataset batched output using TAPETokenizer
    this is required since TAPETokenizer is not compatible to encode the batched output of SequenceDataset
    """

    encoded_sequence_list = []

    for seq in mutation_sequence_batch:
        encoded_sequence_list.append(tokenizer.encode(seq))

    return torch.from_numpy(np.array(encoded_sequence_list)).long()
