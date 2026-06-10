import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from src.baselines.onehot.data import *
from src.baselines.onehot.model import *
from src.baselines.onehot.utils import *
from src.constants import *
from src.utils import *
from torch.utils.data import DataLoader


def collect_onehot_single_mutation_predictions(testing_dataset, onehot_model):
    """
    Collect true labels and predicted scores for one testing dataset.
    Assumes regression / score task.
    """
    testing_dataloader = DataLoader(
        testing_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_function_onehot,
        drop_last=True
    )

    onehot_model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for mutation_sequence_pair_batch, x_batch, label_batch in testing_dataloader:
            logits = onehot_model(x_batch)

            # flatten to scalar
            preds = logits.view(-1).detach().cpu().numpy()
            labels = label_batch.view(-1).detach().cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    return np.array(all_labels), np.array(all_preds)


def nfold_collect_onehot_single_mutation_predictions(saved_dir, data_dir, file_name):
    """
    Collect labels and predictions for each fold.
    
    Returns
    -------
    fold_results : list of dict
        Each entry has:
        {
            "fold": int,
            "labels": np.ndarray,
            "preds": np.ndarray
        }
    """
    result_dict = load_data(os.path.join(saved_dir, "onehot_result_dict.pkl"))
    nfold = result_dict["args"]["nfold"]
    input_dim = len(AA2ID) * len(LACI_WT)

    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold))
    nfold_idx = np.tile(np.arange(nfold), 2)

    fold_results = []

    for n in range(nfold):
        print(f"Processing fold {n}...")

        validation_idx = np.array(nfold_idx[n])
        testing_idx = np.array(nfold_idx[n + 1])
        training_idx = np.array(nfold_idx[n + 2:n + nfold])  # kept for consistency

        testing_data = data_nfold[testing_idx]
        testing_dataset = LacIDatasetOneHot(testing_data, AA2ID, ID2AA)

        onehot_model = SimpleLinear(input_dim, 1)
        onehot_model.load_state_dict(
            torch.load(
                os.path.join(saved_dir, f"onehot_fold{n}_model.pt"),
                map_location="cpu"
            )
        )

        labels, preds = collect_onehot_single_mutation_predictions(
            testing_dataset,
            onehot_model
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
    saving_dir=None
):
    """
    Plot one panel per fold (stacked vertically).

    xticklabels:
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
