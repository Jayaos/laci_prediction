import torch
import torch.nn.functional as F
from itertools import product


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


def _pl_nll_for_order(scores, order):
    """
    Negative log-likelihood of a single PL ranking:
      order: 1D LongTensor of indices in the order y_(1) ≻ y_(2) ≻ ... ≻ y_(m)
    """
    # At each position k, pick item order[k] with prob softmax over remaining items
    loss = scores.new_zeros(())
    remaining = order.clone()
    for k in range(len(order)):
        chosen = remaining[0]                # the item ranked at position k
        den = torch.logsumexp(scores[remaining], dim=0)
        loss = loss - (scores[chosen] - den) # -log softmax
        remaining = remaining[1:]            # remove chosen
        if remaining.numel() == 0:
            break
    return loss


def compute_pl_loss(scores, targets, tie_eps=0.0, reduction="mean",
                    agg_over_ties="mean", max_variants=None):
    """
    Plackett–Luce listwise loss with tie-handling via averaging over tie-breaks.

    Args
    ----
    scores:  (N,) tensor, higher = better (predicted)
    targets: (N,) tensor, higher = better (ground truth)
    tie_eps: items with |t_i - t_j| <= tie_eps are considered tied
    reduction: "mean" | "sum" | "none"
    agg_over_ties: how to combine multiple tie-break permutations: "mean" or "min"
    max_variants: cap the number of tie-break variants (None = enumerate all)

    Behavior
    --------
    1) Sort items by descending target.
    2) Collapse exact/tolerant ties into slots in that order.
    3) For each tie slot with k>1 items, create k variants by selecting one member
       to represent that slot. Form all combinations across slots (Cartesian product).
    4) Compute PL NLL for each variant order (one index per slot), then average (or min).

    Note
    ----
    This follows your example:
      targets = [0, 1, 1, 2], scores = [0.2, 1.2, 1.3, 2.5]
      unique-rank slots = [ {idx of 2}, {idx of 1s (two items)}, {idx of 0} ]
      variants:
        [2, 1a, 0]  → scores [2.5, 1.2, 0.2]
        [2, 1b, 0]  → scores [2.5, 1.3, 0.2]
    """
    device = scores.device
    N = scores.shape[0]
    assert targets.shape == scores.shape

    # 1) sort by descending targets
    order_desc = torch.argsort(targets, dim=0, descending=True)
    t_sorted = targets[order_desc]

    # 2) build tie groups (slots) using tie_eps tolerance
    slots = []
    cur = [order_desc[0].item()]
    for k in range(1, N):
        if torch.abs(t_sorted[k] - t_sorted[k-1]) <= tie_eps:
            cur.append(order_desc[k].item())
        else:
            slots.append(cur)
            cur = [order_desc[k].item()]
    slots.append(cur)

    # If you truly want to collapse to *unique* target values (drop duplicates),
    # this is already achieved by representing each slot with ONE item.

    # 3) enumerate tie-break variants
    # one choice from each slot
    choices_per_slot = [tuple(g) for g in slots]  # list of tuples of item indices
    # total variants = product of sizes
    num_variants = 1
    for g in choices_per_slot:
        num_variants *= len(g)

    # optionally cap variants
    variants = []
    if (max_variants is not None) and (num_variants > max_variants):
        # simple subsample: pick the first element from each slot, then cycle
        # (you can replace with random sampling if preferred)
        # produce up to max_variants variants
        ptrs = [0]*len(choices_per_slot)
        for _ in range(max_variants):
            variant = [choices_per_slot[s][ptrs[s] % len(choices_per_slot[s])]
                       for s in range(len(choices_per_slot))]
            variants.append(torch.tensor(variant, device=device, dtype=torch.long))
            # advance a single pointer (round-robin)
            ptrs[0] += 1
    else:
        for combo in product(*choices_per_slot):
            variants.append(torch.tensor(combo, device=device, dtype=torch.long))

    # 4) compute PL NLL per variant and aggregate
    losses = []
    for ord_idx in variants:
        losses.append(_pl_nll_for_order(scores, ord_idx))
    losses = torch.stack(losses) if len(losses) else scores.new_zeros((1,))

    if agg_over_ties == "mean":
        loss = losses.mean()
    elif agg_over_ties == "min":
        loss = losses.min()
    else:
        raise ValueError("agg_over_ties must be 'mean' or 'min'.")

    if reduction == "mean":
        return loss
    elif reduction == "sum":
        return loss  # already scalar; interpret as sum
    elif reduction == "none":
        return losses  # per-variant losses
    else:
        raise ValueError("reduction must be 'mean' | 'sum' | 'none'")
    

def compute_kl_loss(logits, logits_reg, wt_seq):
    '''
    compute KL regularization loss

    args
    ----
        logits: (batch_size, seq_len, num_tokens)
        logits_reg: 
        wt_seq: encoded wild type sequence batch, (batch_size, seq_len)
    
    returns
    -------

    '''
    creterion_reg = torch.nn.KLDivLoss(reduction='mean')
    batch_size = int(logits.size(0))
    wt_seq_len = wt_seq.size(-1)

    loss = torch.tensor(0.).to(logits.device)
    probs = torch.softmax(logits, dim=-1)
    probs_reg = torch.softmax(logits_reg, dim=-1)

    for i in range(batch_size):

        probs_i = probs[i]
        probs_reg_i = probs_reg[i]

        # idx 1 and the last are the <cls> and <eos>
        reg = probs_reg_i[torch.arange(1, wt_seq_len-1), wt_seq[i, 1:wt_seq_len-1]] # (seq_len,)
        pred = probs_i[torch.arange(1, wt_seq_len-1), wt_seq[i, 1:wt_seq_len-1]] # (seq_len,)

        loss += creterion_reg(reg.log(), pred) # this KL pred * \log( pred / reg)

    return loss / batch_size # avg over batch