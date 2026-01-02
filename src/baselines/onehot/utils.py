import torch



def compute_general_mutation_score(model, x_batch):

    return model(x_batch)


def separate_mutation_sequence_pair(mutation_sequence_pair_batch):
    
    mutations = []
    sequences = []
    
    for ms_pair in mutation_sequence_pair_batch:
        mutations.append(ms_pair[0])
        sequences.append(ms_pair[1])
        
    return mutations, sequences


def get_mutation_position_onehot(mutation_batch):
    """
    obtain the list of mutation position of the sequence
    position in the sequence data dict, wildtype amino acid + position + mutated amino acid, begins with 0.

    args
    ----
        mutation_batch: list of mutations, (batch_size, )
                        for example, [("V0K", "V0L"), ...] for double mutation data

    returns
    -------
        mutation_position: list of position of the mutation. this should only apply to single mutation, (batch_size, 1)
    """

    mutation_position_list = []

    for i in range(len(mutation_batch)):

        mutation_list_i = mutation_batch[i]
        mutation_position_list_i = []
        
        if isinstance(mutation_list_i, str):
            mutation_position_list_i.append(int(mutation_list_i[1:-1]))
            
        if isinstance(mutation_list_i, tuple):
            for mutation in mutation_list_i:
                mutation_position_list_i.append(int(mutation[1:-1]))

        mutation_position_list.append(mutation_position_list_i)

    return torch.tensor(mutation_position_list)


def compute_additive_mutation_score(model, x_batch, mutation_sequence_pair_batch, wt_encoded):
    '''
    compute additive mutation score for SimpleLinear

    args
    ----
        model: SimpleLinear
        x_batch: one-hot encoded mutated sequences, (batch_size, seq_len*20)
        mutation_sequence_pair_batch: 
        wt_encoded: one-hot encoded wild type sequence, (seq_len*20)
    
    returns
    -------
        additive_score: mutational proxy score
    '''

    device = x_batch.device

    mutations, sequences = separate_mutation_sequence_pair(mutation_sequence_pair_batch)

    num_mutations = len(mutations[0])
    batch_size, input_dim = x_batch.size()
    mutation_positions = get_mutation_position_onehot(mutations)
    # (number of mutations, batch_size, seq_len*20)
    decomposed_seq = wt_encoded.clone().unsqueeze(0).unsqueeze(1).expand(num_mutations, batch_size, -1)

    for i in range(batch_size):
        decomposed_seq = decomposed_seq.clone()
        
        for k in range(num_mutations): # iterative over the number of mutations
            decomposed_seq[k, i, 20*mutation_positions[i][k]:20*(mutation_positions[i][k]+1)] = x_batch[i,20*mutation_positions[i][k]:20*(mutation_positions[i][k]+1)]

    decomposed_seq = decomposed_seq.to(device)
    scores = model(decomposed_seq) # (num_mutations, batch_size, 1)
    additive_scores = scores.sum(dim=0)

    return additive_scores # (batch_size, 1)