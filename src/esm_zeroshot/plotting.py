import torch
import matplotlib.pyplot as plt
import esm
from tqdm import tqdm
from src.utils import *
from .data import *
from .utils import *
from src.constants import *


def collect_esm_zeroshot_predictions(
    mutation_data,
    model,
    target_sequence,
    batch_converter,
    random_weight=False,
    device="cpu"
):
    """
    Collect true labels and ESM zero-shot mutation scores.

    Returns
    -------
    labels : np.ndarray
    preds  : np.ndarray
    """
    if random_weight:
        model = init_esm_weight(model)

    data = [("target_sequence", target_sequence)]
    mutation_batch, sequence_batch, x_batch = batch_converter(data)

    model.to(device)
    model.eval()

    with torch.no_grad():
        x_batch = x_batch.to(device)
        outputs = model(x_batch, repr_layers=[model.num_layers - 1], return_contacts=True)
        token_probs = torch.log_softmax(outputs["logits"], dim=-1)

    labels = []
    preds = []

    for mutation_sequence, (mutation, label) in tqdm(mutation_data.items()):
        mutation_score = compute_zeroshot_mutation_score(
            mutation,
            target_sequence,
            token_probs,
            batch_converter.alphabet
        )
        preds.append(float(mutation_score))
        labels.append(float(label))

    return np.array(labels), np.array(preds)


def load_pretrained_esm(pretrained_model):
    """
    Load ESM model and batch_converter from model name.
    """
    if pretrained_model == "esm2_t6_8M_UR50D":
        model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    elif pretrained_model == "esm2_t12_35M_UR50D":
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
    elif pretrained_model == "esm2_t30_150M_UR50D":
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    elif pretrained_model == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    elif pretrained_model == "esm2_t36_3B_UR50D":
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    else:
        raise ValueError("wrong pretrained model input")

    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def nfold_collect_esm_zeroshot_predictions(
    data_dir,
    file_name,
    pretrained_model,
    target_sequence,
    nfold,
    random_weight=False,
    device="cpu"
):
    """
    Collect labels and predictions for each fold for ESM zero-shot evaluation.

    Returns
    -------
    fold_results : list of dict
        [
            {
                "fold": int,
                "labels": np.ndarray,
                "preds": np.ndarray
            },
            ...
        ]
    """
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold))
    nfold_idx = np.tile(np.arange(nfold), 2)

    fold_results = []

    for n in range(nfold):
        print(f"Processing fold {n+1}/{nfold}...")

        validation_idx = np.array(nfold_idx[n])
        testing_idx = np.array(nfold_idx[n + 1])
        training_idx = np.array(nfold_idx[n + 2:n + nfold])

        testing_data = data_nfold[testing_idx]

        print("loading pre-trained ESM2...")
        ESM2_model, alphabet, batch_converter = load_pretrained_esm(pretrained_model)
        print("loading pre-trained ESM2 done")

        labels, preds = collect_esm_zeroshot_predictions(
            testing_data,
            ESM2_model,
            target_sequence,
            batch_converter,
            random_weight=random_weight,
            device=device
        )

        fold_results.append({
            "fold": n,
            "labels": labels,
            "preds": preds
        })

    return fold_results


def plot_nfold_label_vs_prediction_panels(
    fold_results,
    figsize=(7, 20),
    alpha=0.7,
    s=20,
    saving_dir=None,
):
    """
    Plot one panel per fold for ESM zero-shot predictions.

    x-axis tick labels:
    -1 -> nf
     1 -> R1
     2 -> R2
     ...
    10 -> R10
    """
    nfold = len(fold_results)
    fig, axes = plt.subplots(nfold, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    xtick_values = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    xtick_labels = ['NF', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10']

    for ax, fold_result in zip(axes, fold_results):
        fold = fold_result["fold"]
        labels = fold_result["labels"]
        preds = fold_result["preds"]

        ax.scatter(labels, preds, alpha=alpha, s=s)

        ax.set_title(f"Fold {fold+1}")
        ax.set_xlabel("Phenotype")
        ax.set_ylabel("Score")

        ax.set_xticks(xtick_values)
        ax.set_xticklabels(xtick_labels)

        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if saving_dir:
        plt.savefig(saving_dir, bbox_inches="tight")
    plt.show()