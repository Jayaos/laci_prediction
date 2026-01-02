import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from src.utils import *
from tape import TAPETokenizer, UniRepModel
from .data import *
from torchmetrics.classification import BinaryAUROC
from torchmetrics.regression import SpearmanCorrCoef


def nfold_run_unirep_sklearn_linear_head_single_mutation(training_mutation_data_dir,
                                                         training_mutation_file_name,
                                                         testing_mutation_data_dir,
                                                         testing_mutation_file_name,
                                                         representation_data_dir,
                                                         pretrained_model, nfold, task):

    representation_data_dict = load_data(representation_data_dir + pretrained_model + "_single_mutation_representaton_data_dict.pkl")
    training_data_nfold = np.array(load_nfold_data(training_mutation_data_dir, 
                                                   training_mutation_file_name, nfold))
    testing_data_nfold = np.array(load_nfold_data(testing_mutation_data_dir, 
                                                   testing_mutation_file_name, nfold))
    nfold_idx = np.tile(np.arange(nfold),2)

    nfold_result = []

    for n in range(nfold):
        
        print("loading data...")
        validation_idx = np.array(nfold_idx[n])
        testing_idx = np.array(nfold_idx[n+1])
        training_idx = np.array(nfold_idx[n+2:n+nfold])

        # sklearn linear regression doesnt need validation set therefore we include validation set to training set
        training_data = merge_folds(training_data_nfold[np.r_[validation_idx, training_idx]])
        testing_data = testing_data_nfold[testing_idx]

        training_data_array = representation_data_dict_to_array(training_data, representation_data_dict)
        testing_data_array = representation_data_dict_to_array(testing_data, representation_data_dict) 
        print("loading data done")

        model = LinearRegression()
        model.fit(training_data_array[0], training_data_array[1])
        preds = model.predict(testing_data_array[0])

        if task == "binary":
            CRITERION = BinaryAUROC()
            CRITERION.reset()
        elif task == "score":
            CRITERION = SpearmanCorrCoef()
            CRITERION.reset()

        CRITERION.update(torch.tensor(preds).to(torch.float32), 
                         torch.tensor(testing_data_array[1]).to(torch.float32))
        
        nfold_result.append(CRITERION.compute().item())

    if task == "binary":
        print("avg binary AUC: {}".format(np.mean(nfold_result)))
        print("std binary AUC: {}".format(np.std(nfold_result)))
    elif task == "score":
        print("avg spearman corr: {}".format(np.mean(nfold_result)))
        print("std spearman corr: {}".format(np.std(nfold_result)))


def nfold_run_unirep_sklearn_linear_head_multiple_mutation(training_mutation_data_dir,
                                                         training_mutation_file_name,
                                                         training_representation_data_dir,
                                                         testing_mutation_data_dir,
                                                         pretrained_model, nfold, batch_size, task, device):
    """
    train sklearn LinearRegression() on unirep single mutation data
    evaluate the model on unirep multiple mutation
    testing data is generated directly from pretrained UniRep model
    """

    representation_data_dict = load_data(training_representation_data_dir + 
                                         pretrained_model + 
                                         "_single_mutation_representaton_data_dict.pkl")
    training_data_nfold = np.array(load_nfold_data(training_mutation_data_dir, 
                                                   training_mutation_file_name, nfold))
    testing_data = load_data(testing_mutation_data_dir) 
    nfold_idx = np.tile(np.arange(nfold), 2)
    nfold_result = []

    for n in range(nfold):
        
        print("loading data...")
        validation_idx = np.array(nfold_idx[n])
        training_idx = np.array(nfold_idx[n+2:n+nfold])

        # sklearn linear regression doesnt need validation set therefore we include validation set to training set
        training_data = merge_folds(training_data_nfold[np.r_[validation_idx, training_idx]])
        training_data_array = representation_data_dict_to_array(training_data, representation_data_dict)

        print("loading data done")

        # sklearn linear regression model training on single mutation dataset
        model = LinearRegression()
        model.fit(training_data_array[0], training_data_array[1])

        print("loading pre-trained Unirep...")
        if pretrained_model == "unirep-1900":
            embed_dim = 1900
            unirep_model = UniRepModel.from_pretrained("babbler-1900")
            tokenizer = TAPETokenizer(vocab='unirep')
            unirep_model.to(device)
        else:
            raise NotImplementedError("wrong pretrained model input")
        print("loading pre-trained Unirep done")

        test_representations_arr = np.zeros(shape=(len(testing_data), embed_dim))
        test_label = []
        seq_batch = []
        batch_counter = 0

        for mutation_sequence, (mutation, label) in tqdm(testing_data.items()):

            seq_batch.append(mutation_sequence)
            test_label.append(label)

            if len(seq_batch) == batch_size:
                with torch.no_grad():
                    batch_tokens = batch_convert_unirep_sequence(tokenizer, seq_batch)
                    batch_tokens = batch_tokens.to(device)
                    # outputs[0] is hidden representations for all amino acids (batch_size, seq_len+2, embed_dim)
                    # outputs[1] is concatenated representations of the final hidden state and cell state for each sequence,
                    # (batch_size, embed_dim*2)
                    outputs = unirep_model(batch_tokens)
                    # averaged representations over all amino acids in the sequence
                    batch_representations = outputs[0].mean(dim=1) # (batch_size, embed_dim)
                
                test_representations_arr[batch_counter*batch_size:(batch_counter+1)*batch_size] = batch_representations.detach().cpu().numpy()
                batch_counter += 1
                seq_batch = []

        # process the last batch if it's smaller than batch_size
        if len(seq_batch) > 0:
            with torch.no_grad():
                batch_tokens = batch_convert_unirep_sequence(tokenizer, seq_batch)
                batch_tokens = batch_tokens.to(device)
                outputs = unirep_model(batch_tokens)
                batch_representations = outputs[0].mean(dim=1) # (batch_size, embed_dim)

            test_representations_arr[batch_counter*batch_size:] = batch_representations.detach().cpu().numpy()

        test_label_arr = np.array(test_label)

        preds = model.predict(test_representations_arr)

        if task == "binary":
            CRITERION = BinaryAUROC()
            CRITERION.reset()
        elif task == "score":
            CRITERION = SpearmanCorrCoef()
            CRITERION.reset()

        CRITERION.update(torch.tensor(preds).to(torch.float32), 
                         torch.from_numpy(test_label_arr).to(torch.float32))
        print(CRITERION.compute().item())
        nfold_result.append(CRITERION.compute().item())

    if task == "binary":
        print("avg binary AUC: {}".format(np.mean(nfold_result)))
        print("std binary AUC: {}".format(np.std(nfold_result)))
    elif task == "score":
        print("avg spearman corr: {}".format(np.mean(nfold_result)))
        print("std spearman corr: {}".format(np.std(nfold_result)))
