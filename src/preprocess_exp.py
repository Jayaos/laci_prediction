import pandas as pd
import numpy as np
from src.utils import save_data
import copy


# code for addressing reviewers comments


def convert_mutation_to_full_sequence(mutation: str, wildtype_sequence: str):
    """
    convert mutation notation to the corresponding sequence based on the wildtype_sequence

    args
    ----
        mutation: mutation notation, for example "V0K". 
        NOTE: Index starts at 0.
        wildtype_sequence: wildtype protein sequence
    """
    
    wt_residue = mutation[0]
    position = int(mutation[1:-1])
    mt_residue = mutation[-1]
    
    return wildtype_sequence[:position] + mt_residue + wildtype_sequence[(position+1):], wt_residue, mt_residue, position



def WT_heatmap_to_mutation_label_dict_MS(heatmap: pd.DataFrame, sequence: str):
    """
    convert heatmap based on the experimental data to a dictionary
    label only has M vs. S

    args
    ----
        heatmap: heatmap read by using pd.read_csv()
        sequence: target protein sequence

    returns
    -------
        mutation_label_dict: {mutation : label}
    """
    
    mutation_label_dict = dict()
    mutation_candidate = list(heatmap.index)
    
    m_count = 0
    s_count = 0
    misc_count = 0
    
    for i, aa in enumerate(sequence):
        
        try:
            mutation_results = heatmap["{}".format(i+1)] # enumeration index starts at 0, heatmap index starts at 1
        except:
            continue # mutation result starts at the second position, since we ignore the first and the last amino acids 
        
        for mc in mutation_candidate:
            
            mutation_result = mutation_results[mc]
            
            if mutation_result.startswith("M"):
                # non-functional
                label_mutation_result = -1
                m_count += 1
                
            elif mutation_result.startswith("S"):
                # non-functional
                label_mutation_result = 1
                s_count += 1

            else:
                # missing data or wildtype
                label_mutation_result = 0 
                misc_count += 1
            
            mutation = aa + str(i) + mc
            mutation_label_dict[mutation] = label_mutation_result
            
    print("S:{}".format(s_count))
    print("M:{}".format(m_count))
    print("Misc:{}".format(misc_count))
    
    return mutation_label_dict


def WT_heatmap_to_mutation_label_dict_RS(heatmap: pd.DataFrame, sequence: str):
    """
    convert heatmap based on the experimental data to a dictionary
    label only has R vs. S

    args
    ----
        heatmap: heatmap read by using pd.read_csv()
        sequence: target protein sequence

    returns
    -------
        mutation_label_dict: {mutation : label}
    """
    
    mutation_label_dict = dict()
    mutation_candidate = list(heatmap.index)
    
    r_count = 0
    s_count = 0
    misc_count = 0
    
    for i, aa in enumerate(sequence):
        
        try:
            mutation_results = heatmap["{}".format(i+1)] # enumeration index starts at 0, heatmap index starts at 1
        except:
            continue # mutation result starts at the second position, since we ignore the first and the last amino acids 
        
        for mc in mutation_candidate:
            
            mutation_result = mutation_results[mc]
            

            if mutation_result.startswith("S"):
                # non-functional
                label_mutation_result = 1
                s_count += 1
                
            elif mutation_result.startswith("R"):
                # repressor phenotype
                # functional
                label_mutation_result = -1
                r_count += 1
                
            else:
                # missing data or wildtype
                label_mutation_result = 0 
                misc_count += 1
            
            mutation = aa + str(i) + mc
            mutation_label_dict[mutation] = label_mutation_result
            
    print("R:{}".format(r_count))
    print("S:{}".format(s_count))
    print("Misc:{}".format(misc_count))
    
    return mutation_label_dict


def build_sequence_data_dict(heatmap_dir, wildtype_sequence, task, remove_stop_codon=True):
    """
    build sequence data dict using heatmap

    returns
    -------
        sequence_data_dict: {mutated sequence : (mutation, binary label, bin label)}
    """
    heatmap = pd.read_csv(heatmap_dir, index_col=0)
    converted_data = dict()
    stop_codon_count = 0
    missing_count = 0

    if task in ["MS", "ms"]:
        mutation_label_dict = WT_heatmap_to_mutation_label_dict_MS(heatmap, wildtype_sequence)
    elif task in ["RS", "rs"]:
        mutation_label_dict = WT_heatmap_to_mutation_label_dict_RS(heatmap, wildtype_sequence)
    else:
        raise ValueError("wrong task assigned")

    for mutation, label in mutation_label_dict.items():
        full_mutation_sequence, wt_residue, mt_residue, position = convert_mutation_to_full_sequence(mutation, 
                                                                                                     wildtype_sequence)
        
        if remove_stop_codon:
            if mt_residue == "*":
                stop_codon_count += 1
                continue
            elif wt_residue == "*":
                stop_codon_count += 1
                continue
            elif full_mutation_sequence == wildtype_sequence:
                continue
            else:
                # the last and first residue are excluded in WT_heatmap_to_mutation_label_dict()
                converted_data[full_mutation_sequence] = (mutation, label) 

    filtered_data = dict()

    for mutated_sequence, mt_label_pair in converted_data.items():

        mutation, label = mt_label_pair
        
        if label != 0:
            label = int(np.clip(label, 0, 1)) # binary label
            filtered_data[mutated_sequence] = (mt_label_pair[0], label) 
                    
        else:
            # remove data with label 0 (missing data and wildtype)
            missing_count += 1
            continue
            
    print("stop codon count : {}".format(stop_codon_count))
    print("missing count : {}".format(missing_count))
    print("data count : {}".format(len(filtered_data)))
    
    return filtered_data


def split_nfold_single_sequnece_data_dict(binary_data_saving_dir, binary_file_name,
                                          binary_data_dict, nfold, seed=1234):
    """
    two data dict have different number of data
    first split binary sequnece data dict and intersection of each fold with the entire bins data dict is used as
    fold for bins data dict
    """
                
    key_list = list(binary_data_dict.keys())
    np.random.seed(seed)
    np.random.shuffle(key_list)
        
    fold_size = int(np.ceil(len(key_list)/nfold))
        
    for fold in range(nfold):
        
        print("splitting {}-fold".format(fold+1))
        binary_fold_data = dict()
        
        fold_key_list = key_list[fold_size*fold:fold_size*(fold+1)]
        
        for key in fold_key_list:
                binary_fold_data[key] = copy.deepcopy(binary_data_dict[key])
                
                
        print("saving {}-fold of binary data".format(fold+1))
        print("fold size: {}".format(len(binary_fold_data)))
        saving = binary_data_saving_dir + binary_file_name + "_{}fold.pkl".format(fold)
        save_data(saving, binary_fold_data)

