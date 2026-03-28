import numpy as np
import torch
import esm
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.utils import *
from .data import *
from .loss import *
from src.constants import *


def load_pretrained_esm_bt(pretrained_model):
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
    else:
        raise ValueError("wrong pretrained model input")

    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter


def collect_bt_predictions(testing_dataset, model, batch_converter, device):
    """
    Collect true labels and predicted mutation scores for one fold.

    Returns
    -------
    labels : np.ndarray
    preds  : np.ndarray
    """
    testing_dataloader = DataLoader(
        testing_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_function_sequence_dataset,
        drop_last=False
    )

    model.to(device)
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for mutation_sequence_pair_batch, label_batch in tqdm(testing_dataloader):
            mutation_batch, sequence_batch, x_batch = batch_converter(mutation_sequence_pair_batch)
            _, _, wt_batch = batch_converter((("wt", LACI_WT),))
            wt_batch = wt_batch.repeat(x_batch.size(0), 1)

            x_batch = x_batch.to(device)
            wt_batch = wt_batch.to(device)
            label_batch = label_batch.to(dtype=torch.float32).to(device)

            mutation_scores, logits = compute_mutation_score(
                model, x_batch, wt_batch, mutation_batch, batch_converter
            )

            all_preds.extend(mutation_scores.view(-1).detach().cpu().numpy().tolist())
            all_labels.extend(label_batch.view(-1).detach().cpu().numpy().tolist())

    return np.array(all_labels), np.array(all_preds)


def nfold_collect_bt_predictions(
    data_dir,
    saving_dir,
    file_name,
    pretrained_model,
    nfold,
    device
):
    """
    Load each saved fine-tuned BT fold model and collect labels/predictions.

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
        testing_dataset = SequenceDataset(testing_data)

        print("loading pre-trained ESM2 backbone...")
        model, alphabet, batch_converter = load_pretrained_esm_bt(pretrained_model)

        model_path = saving_dir + f"ESM2_bt_fold{n}_model.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("model loading done")

        labels, preds = collect_bt_predictions(
            testing_dataset=testing_dataset,
            model=model,
            batch_converter=batch_converter,
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
    s=20
):
    """
    Plot one panel per fold for ESM-BT model.

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
    xtick_labels = ["NF", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]

    for ax, fold_result in zip(axes, fold_results):
        fold = fold_result["fold"]
        labels = fold_result["labels"]
        preds = fold_result["preds"]

        ax.scatter(labels, preds, alpha=alpha, s=s)

        ax.set_xlabel("Label score")
        ax.set_ylabel("Predicted mutation score")

        ax.set_xticks(xtick_values)
        ax.set_xticklabels(xtick_labels)
        ax.set_xlim(-1.5, 10.5)

        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()