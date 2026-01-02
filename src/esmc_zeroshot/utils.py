

def compute_zeroshot_mutation_score(mutation, wt_sequence, token_probs, tokenizer):
    # split mutation into wildtype_residue, position, mutation_residue

    if isinstance(mutation, str):
        # single mutation
        wt_aa, pos, mt_aa = mutation[0], int(mutation[1:-1]), mutation[-1]

        if wt_sequence[pos] != wt_aa:
            raise ValueError("The listed wildtype does not match the provided sequence")

        wt_encoded, mt_encoded = tokenizer.convert_tokens_to_ids(wt_aa), tokenizer.convert_tokens_to_ids(mt_aa)

        # the first position in the sequence is for <CLS>
        # log-likelihood of mutation - log-likelihood of wildtype
        score = token_probs[0, pos+1, mt_encoded] - token_probs[0, pos+1, wt_encoded]
        score = score.item()

    elif isinstance(mutation, tuple):
        # multiple mutation

        score = 0

        for mt in mutation:
            wt_aa, pos, mt_aa = mt[0], int(mt[1:-1]), mt[-1]
            
            if wt_sequence[pos] != wt_aa:
                raise ValueError("The listed wildtype does not match the provided sequence")

            wt_encoded, mt_encoded = tokenizer.convert_tokens_to_ids(wt_aa), tokenizer.convert_tokens_to_ids(mt_aa)

            # the first position in the sequence is for <CLS>
            # log-likelihood of mutation - log-likelihood of wildtype
            this_score = token_probs[0, pos+1, mt_encoded] - token_probs[0, pos+1, wt_encoded]
            score += this_score.item()
        
    return score