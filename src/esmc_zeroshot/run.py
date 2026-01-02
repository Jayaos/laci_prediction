import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig
from tqdm import tqdm
from src.utils import *
from .utils import *
from src.constants import *
from torchmetrics.classification import BinaryAUROC
from torchmetrics.regression import SpearmanCorrCoef



def nfold_evaluate_zeroshot_single_mutation(data_dir, file_name, pretrained_model, target_sequence, nfold, task, device):
    """
    n-fold evaluation of ESMC model on the given task using zero-shot mutation score
    """

    print("loading pre-trained ESMC...")
    if pretrained_model == "300m":
        ESMC_model = ESMC.from_pretrained("esmc_300m")
    elif pretrained_model == "600m":
        ESMC_model = ESMC.from_pretrained("esmc_600m")
    else:
        raise ValueError("wrong pretrained model input")
    print("loading pre-trained ESMC done")

    test_metric_list = []
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)

    for n in range(nfold):
        
        print("loading data...")
        validation_idx = np.array(nfold_idx[n])
        testing_idx = np.array(nfold_idx[n+1])
        training_idx = np.array(nfold_idx[n+2:n+nfold])
        testing_data = data_nfold[testing_idx] # zero-shot prediction does not use SequenceDataset
        print("loading data done")

        test_metric = evaluate_esmc_zeroshot(testing_data, ESMC_model, target_sequence, task, device)
        test_metric_list.append(test_metric)

    if task == "binary":
        print("avg binary AUC: {}".format(np.mean(test_metric_list)))
        print("std binary AUC: {}".format(np.std(test_metric_list)))
    elif task == "score":
        print("avg spearman corr: {}".format(np.mean(test_metric_list)))
        print("std spearman corr: {}".format(np.std(test_metric_list)))
    
    return test_metric_list


def evaluate_esmc_zeroshot(mutation_data, model, target_sequence, task, device="cpu"):
    
    if task == "binary":
        CRITERION = BinaryAUROC()
        CRITERION.reset()
    elif task == "score":
        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()
    
    # tokenizer adds <CLS> token at the first position
    # and <EOS> token at the last position
    protein = ESMProtein(sequence=target_sequence)
    x = model.encode(protein)
    
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        outputs = model.logits(x, LogitsConfig(sequence=True, return_embeddings=True))
        token_probs = torch.log_softmax(outputs.logits.sequence, dim=-1)
        
    for mutation_sequence, (mutation, label) in tqdm(mutation_data.items()):

        mutation_score = compute_zeroshot_mutation_score(mutation, target_sequence, token_probs, model.tokenizer)
        CRITERION.update(torch.tensor([mutation_score]), torch.tensor([label]).to(torch.float32))
    
    return CRITERION.compute().item()


def evaluate_zeroshot_multiple_mutations(data_dir, pretrained_model, task, device):
    """
    evaluation of ESM2 model on the given task using zero-shot mutation score
    """

    print("loading data...")
    testing_data = load_data(data_dir)
    print("loading data done")

    print("loading pre-trained ESMC...")
    if pretrained_model == "300m":
        ESMC_model = ESMC.from_pretrained("esmc_300m")
    elif pretrained_model == "600m":
        ESMC_model = ESMC.from_pretrained("esmc_600m")
    else:
        raise ValueError("wrong pretrained model input")
    print("loading pre-trained ESMC done")

    test_metric = evaluate_esmc_zeroshot(testing_data, ESMC_model, task, device)

    if task == "binary":
        print("binary AUC: {}".format(test_metric))
    elif task == "score":
        print("spearman corr: {}".format(test_metric))
    
    return test_metric