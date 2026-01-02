import torch
import torch.nn.functional as F


def compute_mutation_score(model, mt_seq, wt_seq, mutation, batch_converter):
    '''
    compute masked marginal score

    NOTE about mutation position
    batch_converter adds <cls> token at the first position and <eos> token at the last position
    position in the sequence data dict, wildtype amino acid + position + mutated amino acid, begins with 0.
    Therefore, we need to adjust the insertion of the <cls> token positioned at the first position.

    args
    ----
        model: ESM2 model
        mt_seq: encoded mutated sequences, (batch_size, seq_len)
        wt_seq: encoded wild type sequence, (batch_size, seq_len)
        mutation: list of tuples containing strings of mutantation e.g., "A8K", (batch_size, tuples of mutations)
        batch_converter: batch converter object
    
    returns
    -------
        score: mutational proxy score
        logits: output logits for masked sequence
    '''
    device = mt_seq.device

    masked_seq = mt_seq.clone()
    mask_idx = batch_converter.alphabet.mask_idx

    batch_size = int(mt_seq.size(0))
    scores = torch.zeros(batch_size).to(device)

    for i in range(batch_size):

        if isinstance(mutation[i], str):
            # single mutation
            mt_pos = int(mutation[i][1:-1]) + 1 # refer to NOTE about + 1
        elif isinstance(mutation[i], tuple):
            # multiple mutations
            mt_pos = []

            for k in range(len(mutation[i])):
                mt_pos.append(int(mutation[i][k][1:-1]) + 1) # refer to NOTE about + 1

        masked_seq[i, mt_pos] = mask_idx

    masked_seq = masked_seq.to(device)

    output = model(masked_seq, return_contacts=False)
    logits = output["logits"] # (batch_size, seq_len, num_tokens)
    log_probs = torch.log_softmax(logits, dim=-1) # log_softmax along the dimension of tokens

    for i in range(batch_size):

        if isinstance(mutation[i], str):
            # single mutation
            mt_pos = int(mutation[i][1:-1]) + 1 # refer to NOTE about + 1
        elif isinstance(mutation[i], tuple):
            # multiple mutations
            mt_pos = []

            for k in range(len(mutation[i])):
                mt_pos.append(int(mutation[i][k][1:-1]) + 1) # refer to NOTE about + 1

        score_i = log_probs[i] # (seq_len, num_tokens)
        mt_seq_i = mt_seq[i] # (seq_len)
        wt_seq_i = wt_seq[i] # (seq_len)
        # computation of the masked marginal probability as in Meier et al. 2021
        # \sum_{i \in M} \log p(x_i = x_i^{\text{mt}} | x_{-M}) - \log p(x_i = x_i^{\text{wt}} | x_{-M})
        scores[i] = torch.sum(score_i[mt_pos, mt_seq_i[mt_pos]]) - \
            torch.sum(score_i[mt_pos, wt_seq_i[mt_pos]])

    return scores, logits


def compute_BT_loss(scores, target_scores):
    """
    compute Bradely-Terry loss

    args
    ----
        scores: , (batch_size)
        target_scores: , (batch_size)

    """

    loss = torch.tensor(0.).to(scores.device)

    for i in range(len(scores)):
        for j in range(i, len(scores)):

            if target_scores[i] > target_scores[j]:
                loss += torch.log(1+torch.exp(scores[j]-scores[i]))
            else:
                loss += torch.log(1+torch.exp(scores[i]-scores[j]))

    return loss


def compute_BT_loss_(scores, targets, tie_eps=0.0, reduction="mean"):
    """
    Bradley-Terry pairwise ranking loss.
      scores:  (N,) model scores (higher = better)
      targets: (N,) ground-truth scalar fitness (higher = better)
      tie_eps: skip pairs with |targets_i - targets_j| <= tie_eps
      reduction: "mean" | "sum" | "none"
    """
    device = scores.device

    N = scores.shape[0]
    idx = torch.arange(N, device=device)
    pairs = torch.combinations(idx, r=2)            # shape [N, 2]
    i, j = pairs[:,0], pairs[:,1]

    # excluding ties
    diff_t = targets[i] - targets[j]
    mask_pos = diff_t >  tie_eps  
    mask_neg = diff_t < -tie_eps   

    # For i ≻ j, penalize if score_i ≤ score_j: softplus(score_j - score_i)
    loss_pos = F.softplus(scores[j][mask_pos] - scores[i][mask_pos])

    # For j ≻ i, penalize if score_j ≤ score_i: softplus(score[i] - score[j])
    loss_neg = F.softplus(scores[i][mask_neg] - scores[j][mask_neg])

    loss = torch.cat([loss_pos, loss_neg], dim=0)
    if reduction == "mean":
        return loss.mean() if loss.numel() else torch.zeros((), device=device)
    if reduction == "sum":
        return loss.sum()
    return loss


def compute_KL_loss(logits, logits_reg, wt_seq):
    '''
    compute KL regularization loss

    args
    ----
        model: 
        wt_seq: encoded wild type sequence batch, (batch_size, seq_len)
    
    returns
    -------

    '''
    creterion_reg = torch.nn.KLDivLoss(reduction='batchmean')
    batch_size = int(logits.size(0))
    wt_seq_len = wt_seq.size(-1)

    loss = torch.tensor(0.).to(logits.device)
    probs = torch.softmax(logits, dim=-1)
    probs_reg = torch.softmax(logits_reg, dim=-1)

    for i in range(batch_size):

        probs_i = probs[i]
        probs_reg_i = probs_reg[i]

        # idx 1 and the last are the <cls> and <eos>
        reg = probs_reg_i[torch.arange(1, wt_seq_len-1), wt_seq[i, 1:wt_seq_len-1]]
        pred = probs_i[torch.arange(1, wt_seq_len-1), wt_seq[i, 1:wt_seq_len-1]]

        loss += creterion_reg(reg.log(), pred)

    return loss