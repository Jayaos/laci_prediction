import torch
import numpy as np
from tqdm import tqdm
from src.baselines.onehot.data import *
from src.baselines.onehot.model import *
from src.baselines.onehot.utils import *
from src.constants import *
from src.utils import *
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics.regression import SpearmanCorrCoef


def train_onehot(training_dataset, validation_dataset, input_dim, max_epoch, batch_size, learning_rate, 
                 train_objective, early_stop):
    
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                                     collate_fn=collate_function_onehot, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, 
                                       collate_fn=collate_function_onehot, drop_last=True)

    print("building and initializing model...")
    onehot_model = SimpleLinear(input_dim, 1)
    
    if train_objective == "bce":
        Sigmoid = torch.nn.Sigmoid()
        loss_fn = torch.nn.BCELoss()
    elif train_objective == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError("wrong objective assigned")

    Optimizer = torch.optim.Adam(onehot_model.parameters(), lr=learning_rate)
    print("building and initializing model done...")

    print("training starts...")
    training_loss_record = []
    validation_loss_record = []
    validation_metric_record = []
    best_metric = 0
    best_model = onehot_model.state_dict()
    best_epoch = 0


    batch_loss_sum = 0.
    num_total_batch = len(validation_dataloader)
    onehot_model.eval()
    for mutation_sequence_pair_batch, x_batch, label_batch in tqdm(validation_dataloader):
        
        with torch.no_grad():
            logits = onehot_model(x_batch) # batch_size * num_class
            label_batch = label_batch.view(logits.size())

            if train_objective == "bce":
                loss_batch = loss_fn(Sigmoid(logits), label_batch)
            elif train_objective == "mse":
                loss_batch = loss_fn(logits, label_batch)

            batch_loss_sum += loss_batch.item()
        
    validation_loss = batch_loss_sum/num_total_batch
    validation_loss_record.append(validation_loss)
    print("initial validation loss: {}".format(validation_loss))

    for e in range(max_epoch):

        if early_stop != None:
            if (e+1-best_epoch) > early_stop:
                # if the loss did not decrease for pre-defined number of epoch in a row, stop training
                break

        print("epoch {}".format(e+1))
        batch_loss_sum = 0.
        num_total_batch = len(training_dataloader)

        onehot_model.train()
        for mutation_sequence_pair_batch, x_batch, label_batch in tqdm(training_dataloader):
            
            Optimizer.zero_grad()
            logits = onehot_model(x_batch) # batch_size * num_class
            label_batch = label_batch.view(logits.size())

            if train_objective == "bce":
                loss_batch = loss_fn(Sigmoid(logits), label_batch)
            elif train_objective == "mse":
                loss_batch = loss_fn(logits, label_batch)

            loss_batch.backward()
            Optimizer.step()
            batch_loss_sum += loss_batch.item()
        
        training_loss = batch_loss_sum/num_total_batch
        training_loss_record.append(training_loss)
        print("training loss: {}".format(training_loss))

        print("calculating validation loss...")
        batch_loss_sum = 0.
        num_total_batch = len(validation_dataloader)

        if train_objective == "bce":
            CRITERION = BinaryAUROC()
            CRITERION.reset()
        elif train_objective == "mse":
            # no activation
            CRITERION = SpearmanCorrCoef()
            CRITERION.reset()

        onehot_model.eval()
        for mutation_sequence_pair_batch, x_batch, label_batch in tqdm(validation_dataloader):

            with torch.no_grad():
                logits = onehot_model(x_batch) # batch_size * num_class
                label_batch = label_batch.view(logits.size())

                if train_objective == "bce":
                    loss_batch = loss_fn(Sigmoid(logits), label_batch)
                    CRITERION.update(Sigmoid(logits), label_batch)
                elif train_objective == "mse":
                    loss_batch = loss_fn(logits, label_batch)
                    CRITERION.update(logits, label_batch)
            
                batch_loss_sum += loss_batch.item()

        validation_loss = batch_loss_sum/num_total_batch
        validation_loss_record.append(validation_loss)
        validation_metric = CRITERION.compute().item()
        validation_metric_record.append(validation_metric)

        if validation_metric > best_metric:
            best_metric = validation_metric
            best_epoch = e+1
            best_model = onehot_model.state_dict()

        print("training loss: {}".format(training_loss))
        print("validation loss: {}".format(validation_loss))

        if train_objective == "bce":
            print("validation AUC: {}".format(validation_metric))
        elif train_objective == "mse":
            print("validation spearman corr: {}".format(validation_metric))

    print("best model at epoch {}".format(best_epoch))
    print("saving results...")
    result_dict = {"training_epoch" : e+1, "batch_size" : batch_size, "learning_rate" : learning_rate, 
                   "training_loss_record" : training_loss_record, "validation_loss_record" : validation_loss_record,
                   "validation_metric_record" : validation_metric_record}
    
    onehot_model.load_state_dict(best_model)

    return onehot_model, result_dict


def evaluate_onehot_single_mutation(testing_dataset, onehot_model, train_objective, task):

    testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=True, 
                                       collate_fn=collate_function_onehot, drop_last=True)
    
    if task == "binary":
        Sigmoid = torch.nn.Sigmoid()
        CRITERION = BinaryAUROC()
        CRITERION.reset()
    elif task == "score":
        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()

    onehot_model.eval()
    for mutation_sequence_pair_batch, x_batch, label_batch in tqdm(testing_dataloader):

            with torch.no_grad():
                logits = onehot_model(x_batch) 
                label_batch = label_batch.view(logits.size())

                if task == "binary" and train_objective == "bce":
                    CRITERION.update(Sigmoid(logits), label_batch)
                elif task == "binary" and train_objective == "mse":
                    CRITERION.update(logits, label_batch)
                elif task == "score" and train_objective == "mse":
                    CRITERION.update(logits, label_batch)
                else:
                    raise ValueError("wrong task and train objective assigned")
        
    return CRITERION.compute().item()


def evaluate_onehot_multiple_mutations(testing_dataset, onehot_model, score_computation, train_objective, task):

    testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=True, 
                                       collate_fn=collate_function_onehot, drop_last=True)
    # one-hot encoded wild type sequence
    wt_encoded = torch.from_numpy(np.eye(len(AA2ID))[[AA2ID[aa] for aa in LACI_WT]]).flatten().to(torch.float32)
    
    if task == "binary":
        Sigmoid = torch.nn.Sigmoid()
        CRITERION = BinaryAUROC()
        CRITERION.reset()
    elif task == "score":
        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()

    onehot_model.eval()
    for mutation_sequence_pair_batch, x_batch, label_batch in tqdm(testing_dataloader):

            with torch.no_grad():

                if task == "binary" and train_objective == "bce":
                    # no additive score with BCE train objective and binary prediction task
                    if score_computation == "general":
                        logits = onehot_model(x_batch)
                        mutation_scores = Sigmoid(logits)
                    else:
                        raise ValueError("wrong score computation method")
                    
                elif task == "binary" and train_objective == "mse":
                    if score_computation == "general":
                        mutation_scores = compute_general_mutation_score(onehot_model, x_batch)
                    elif score_computation == "additive":
                        mutation_scores = compute_additive_mutation_score(onehot_model, x_batch,
                                                                          mutation_sequence_pair_batch,
                                                                          wt_encoded)
                        
                elif task == "score" and train_objective == "mse":
                    if score_computation == "general":
                        mutation_scores = compute_general_mutation_score(onehot_model, x_batch)
                    elif score_computation == "additive":
                        mutation_scores = compute_additive_mutation_score(onehot_model, x_batch,
                                                                          mutation_sequence_pair_batch,
                                                                          wt_encoded)
                else:
                    raise ValueError("wrong task and train objective assigned")
                
                label_batch = label_batch.view(mutation_scores.size())
                CRITERION.update(mutation_scores, label_batch)
        
    return CRITERION.compute().item()

    
def nfold_train_onehot(data_dir, saving_dir, file_name, nfold, max_epoch, batch_size, 
                       learning_rate, train_objective, early_stop):
    
    nfold_result_dict = dict()
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)
    input_dim = len(AA2ID)*len(LACI_WT)

    for n in range(nfold):
        
        print("loading data...")
        validation_idx = np.array(nfold_idx[n])
        testing_idx = np.array(nfold_idx[n+1])
        training_idx = np.array(nfold_idx[n+2:n+nfold])

        training_data = merge_folds(data_nfold[training_idx])
        validation_data = data_nfold[validation_idx]
        testing_data = data_nfold[testing_idx]

        training_dataset = LacIDatasetOneHot(training_data, AA2ID, ID2AA)
        validation_dataset = LacIDatasetOneHot(validation_data, AA2ID, ID2AA)
        testing_dataset = LacIDatasetOneHot(testing_data, AA2ID, ID2AA)
        print("loading data done")

        best_model, result_dict = train_onehot(training_dataset, validation_dataset, input_dim, max_epoch, batch_size, 
                                               learning_rate, train_objective, early_stop)        
        torch.save(best_model.state_dict(), saving_dir+"onehot_fold{}_model.pt".format(n))
        nfold_result_dict[n] = result_dict
        
    # saving arguments of the function
    args = locals()
    nfold_result_dict["args"] = args

    save_data(saving_dir+"onehot_result_dict.pkl", nfold_result_dict)


def nfold_evaluate_onehot_single_mutation(saved_dir, data_dir, file_name, task):
    """
    evaluate n-fold saved models on the given task
    """
    result_dict = load_data(saved_dir + "onehot_result_dict.pkl")
    nfold = result_dict["args"]["nfold"]
    input_dim = len(AA2ID)*len(LACI_WT)
    
    test_metric_list = []
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)

    for n in range(nfold):
        
        print("loading data...")
        validation_idx = np.array(nfold_idx[n])
        testing_idx = np.array(nfold_idx[n+1])
        training_idx = np.array(nfold_idx[n+2:n+nfold])

        testing_data = data_nfold[testing_idx]
        testing_dataset = LacIDatasetOneHot(testing_data, AA2ID, ID2AA)
        print("loading data done")

        onehot_model = SimpleLinear(input_dim, 1)
        onehot_model.load_state_dict(torch.load(saved_dir + "onehot_fold{}_model.pt".format(n)))
        test_metric = evaluate_onehot_single_mutation(testing_dataset, onehot_model, 
                                                      result_dict["args"]["train_objective"], 
                                                      task)
        test_metric_list.append(test_metric)

    if task == "binary":
        print("avg binary AUC: {}".format(np.mean(test_metric_list)))
        print("std binary AUC: {}".format(np.std(test_metric_list)))
    elif task == "score":
        print("avg spearman corr: {}".format(np.mean(test_metric_list)))
        print("std spearman corr: {}".format(np.std(test_metric_list)))
    
    return test_metric_list


def nfold_evaluate_onehot_multiple_mutations(saved_dir, data_dir, score_computation, task):
    """
    evaluate n-fold saved models on the given task
    """
    result_dict = load_data(saved_dir + "onehot_result_dict.pkl")
    nfold = result_dict["args"]["nfold"]
    input_dim = len(AA2ID)*len(LACI_WT)
    
    test_metric_list = []
    print("loading data...")
    testing_data = load_data(data_dir)
    testing_dataset = LacIDatasetOneHot(testing_data, AA2ID, ID2AA)
    print("loading data done")

    for n in range(nfold):

        onehot_model = SimpleLinear(input_dim, 1)
        onehot_model.load_state_dict(torch.load(saved_dir + "onehot_fold{}_model.pt".format(n)))
        test_metric = evaluate_onehot_multiple_mutations(testing_dataset, onehot_model, score_computation,
                                                      result_dict["args"]["train_objective"], task)
        test_metric_list.append(test_metric)

    if task == "binary":
        print("avg binary AUC: {}".format(np.mean(test_metric_list)))
        print("std binary AUC: {}".format(np.std(test_metric_list)))
    elif task == "score":
        print("avg spearman corr: {}".format(np.mean(test_metric_list)))
        print("std spearman corr: {}".format(np.std(test_metric_list)))
    
    return test_metric_list


# code for addressing reviewers comments
def evaluate_onehot_single_mutation_exp(testing_dataset, onehot_model):

    testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=True, 
                                       collate_fn=collate_function_onehot, drop_last=True)
    
    Sigmoid = torch.nn.Sigmoid()
    AUROC_CRITERION = BinaryAUROC()
    AUROC_CRITERION.reset()
    PRAUC_CRITERION = BinaryAveragePrecision()
    PRAUC_CRITERION.reset()
    num_label_1 = 0
    num_label_0 = 0

    estimated = []
    labels = []

    onehot_model.eval()
    for mutation_sequence_pair_batch, x_batch, label_batch in tqdm(testing_dataloader):

            with torch.no_grad():
                logits = onehot_model(x_batch) 
                label_batch = label_batch.view(logits.size())
                num_label_1 += (label_batch == 1).sum().item()
                num_label_0 += (label_batch == 0).sum().item()
                label_batch = label_batch.long()
                probs = Sigmoid(logits)
                AUROC_CRITERION.update(probs, label_batch)
                PRAUC_CRITERION.update(probs, label_batch)
                estimated.append(probs.item())
                labels.append(label_batch.item())
        
    return AUROC_CRITERION.compute().item(), PRAUC_CRITERION.compute().item(), \
        num_label_1 / (num_label_1+num_label_0), estimated, labels


def nfold_evaluate_onehot_single_mutation_exp(saved_dir, data_dir, file_name):
    """
    evaluate n-fold saved models
    """
    result_dict = load_data(saved_dir + "onehot_result_dict.pkl")
    nfold = result_dict["args"]["nfold"]
    input_dim = len(AA2ID)*len(LACI_WT)
    
    auroc_list = []
    prauc_list = []
    label_ratio_list = []
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)
    probs_list = []
    labels_list = []

    for n in range(nfold):
        
        print("loading data...")
        validation_idx = np.array(nfold_idx[n])
        testing_idx = np.array(nfold_idx[n+1])
        training_idx = np.array(nfold_idx[n+2:n+nfold])

        testing_data = data_nfold[testing_idx]
        testing_dataset = LacIDatasetOneHot(testing_data, AA2ID, ID2AA)
        print("loading data done")

        onehot_model = SimpleLinear(input_dim, 1)
        onehot_model.load_state_dict(torch.load(saved_dir + "onehot_fold{}_model.pt".format(n)))
        auroc, prauc, label_ratio, probs, labels = evaluate_onehot_single_mutation_exp(testing_dataset, onehot_model)
        auroc_list.append(auroc)
        prauc_list.append(prauc)
        label_ratio_list.append(label_ratio)
        probs_list.append(probs)
        labels_list.append(labels)

    print("avg AUC: {}".format(np.mean(auroc_list)))
    print("std AUC: {}".format(np.std(auroc_list)))
    print("avg PRAUC: {}".format(np.mean(prauc_list)))
    print("std PRAUC: {}".format(np.std(prauc_list)))
    print("avg label ratio: {}".format(np.mean(label_ratio_list)))

    return probs_list, labels_list
