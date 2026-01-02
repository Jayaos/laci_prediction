



def batch_convert_unirep(tokenizer: TAPETokenizer, mutation_sequence_pair_batch):
    """
    encode SequenceDataset batched output using TAPETokenizer
    this is required since TAPETokenizer is not compatible to encode the batched output of SequenceDataset
    """

    encoded_sequence_list = []

    for mut, seq in mutation_sequence_pair_batch:
        encoded_sequence_list.append(tokenizer.encode(seq))

    return torch.from_numpy(np.array(encoded_sequence_list)).long()