import torch
import numpy as np
from tqdm import tqdm
from .data import *
from .model import *
from .utils import compute_general_mutation_score, compute_additive_mutation_score
from src.constants import *
from src.utils import *
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC
from torchmetrics.regression import SpearmanCorrCoef


def train_mlp(training_dataset, validation_dataset, 
              input_dim, hidden_dim, max_epoch, batch_size, learning_rate, 
              train_objective, early_stop, device):
    
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                                     collate_fn=collate_function_onehot, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, 
                                       collate_fn=collate_function_onehot, drop_last=True)

    print("building and initializing model...")
    mlp_model = MLP(input_dim, hidden_dim, 1)
    mlp_model.to(device)
    
    if train_objective == "bce":
        Sigmoid = torch.nn.Sigmoid()
        loss_fn = torch.nn.BCELoss()
    elif train_objective == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError("wrong objective assigned")

    Optimizer = torch.optim.Adam(mlp_model.parameters(), lr=learning_rate)
    print("building and initializing model done...")

    print("training starts...")
    training_loss_record = []
    validation_loss_record = []
    validation_metric_record = []
    best_metric = 0
    best_model = mlp_model.state_dict()
    best_epoch = 0


    batch_loss_sum = 0.
    num_total_batch = len(validation_dataloader)
    mlp_model.eval()
    for mutation_sequence_pair_batch, x_batch, label_batch in tqdm(validation_dataloader):
        
        with torch.no_grad():
            x_batch = x_batch.to(device)
            logits = mlp_model(x_batch)
            label_batch = label_batch.view(logits.size())
            label_batch = label_batch.to(device)

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

        mlp_model.train()
        for mutation_sequence_pair_batch, x_batch, label_batch in tqdm(training_dataloader):
            
            Optimizer.zero_grad()
            x_batch = x_batch.to(device)
            logits = mlp_model(x_batch) 
            label_batch = label_batch.view(logits.size())
            label_batch = label_batch.to(device)

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

        mlp_model.eval()
        for mutation_sequence_pair_batch, x_batch, label_batch in tqdm(validation_dataloader):

            with torch.no_grad():
                x_batch = x_batch.to(device)
                logits = mlp_model(x_batch) # batch_size * num_class
                label_batch = label_batch.view(logits.size())
                label_batch = label_batch.to(device)

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
            best_model = mlp_model.state_dict()

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
    
    mlp_model.load_state_dict(best_model)

    return mlp_model, result_dict


def evaluate_mlp_single_mutation(testing_dataset, mlp_model, train_objective, task, device):

    testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=True, 
                                       collate_fn=collate_function_onehot, drop_last=True)
    
    if task == "binary":
        Sigmoid = torch.nn.Sigmoid()
        CRITERION = BinaryAUROC()
        CRITERION.reset()
    elif task == "score":
        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()

    mlp_model.to(device)

    mlp_model.eval()
    for mutation_sequence_pair_batch, x_batch, label_batch in tqdm(testing_dataloader):

            with torch.no_grad():
                x_batch = x_batch.to(device)
                logits = mlp_model(x_batch) 
                label_batch = label_batch.view(logits.size())
                label_batch = label_batch.to(device)

                if task == "binary" and train_objective == "bce":
                    CRITERION.update(Sigmoid(logits), label_batch)
                elif task == "binary" and train_objective == "mse":
                    CRITERION.update(logits, label_batch)
                elif task == "score" and train_objective == "mse":
                    CRITERION.update(logits, label_batch)
                else:
                    raise ValueError("wrong task and train objective assigned")
        
    return CRITERION.compute().item()


def evaluate_mlp_multiple_mutations(testing_dataset, mlp_model, score_computation, train_objective, task, device):

    testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=True, 
                                       collate_fn=collate_function_onehot, drop_last=True)
    # one-hot encoded wild type sequence
    wt_encoded = torch.from_numpy(np.eye(len(AA2ID))[[AA2ID[aa] for aa in LACI_WT]]).flatten().to(torch.float32)
    
    if task == "binary":
        sigmoid = torch.nn.Sigmoid()
        CRITERION = BinaryAUROC()
        CRITERION.reset()
    elif task == "score":
        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()

    mlp_model.to(device)

    mlp_model.eval()
    for mutation_sequence_pair_batch, x_batch, label_batch in tqdm(testing_dataloader):

            with torch.no_grad():

                x_batch = x_batch.to(device)

                if task == "binary" and train_objective == "bce":
                    # no additive score with BCE train objective and binary prediction task
                    if score_computation == "general":
                        logits = mlp_model(x_batch)
                        mutation_scores = sigmoid(logits)
                    else:
                        raise ValueError("wrong score computation method")
                    
                elif task == "binary" and train_objective == "mse":
                    if score_computation == "general":
                        mutation_scores = compute_general_mutation_score(mlp_model, x_batch)
                    elif score_computation == "additive":
                        mutation_scores = compute_additive_mutation_score(mlp_model, x_batch,
                                                                          mutation_sequence_pair_batch,
                                                                          wt_encoded)
                        
                elif task == "score" and train_objective == "mse":
                    if score_computation == "general":
                        mutation_scores = compute_general_mutation_score(mlp_model, x_batch)
                    elif score_computation == "additive":
                        mutation_scores = compute_additive_mutation_score(mlp_model, x_batch,
                                                                          mutation_sequence_pair_batch,
                                                                          wt_encoded)
                else:
                    raise ValueError("wrong task and train objective assigned")
                
                label_batch = label_batch.view(mutation_scores.size())
                label_batch = label_batch.to(device)
                CRITERION.update(mutation_scores, label_batch)
        
    return CRITERION.compute().item()


def nfold_train_mlp(data_dir, saving_dir, file_name, nfold, hidden_dim, max_epoch, batch_size, 
                    learning_rate, train_objective, early_stop, device):
    
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

        # MLP uses the same dataset class for onehot
        training_dataset = LacIDatasetOneHot(training_data, AA2ID, ID2AA)
        validation_dataset = LacIDatasetOneHot(validation_data, AA2ID, ID2AA)
        print("loading data done")

        best_model, result_dict = train_mlp(training_dataset, validation_dataset, 
                                            input_dim, hidden_dim , max_epoch, batch_size, 
                                            learning_rate, train_objective, early_stop, device)        
        torch.save(best_model.state_dict(), saving_dir+"mlp_fold{}_model.pt".format(n))
        nfold_result_dict[n] = result_dict

    # saving arguments of the function
    args = locals()
    nfold_result_dict["args"] = args

    save_data(saving_dir+"mlp_result_dict.pkl", nfold_result_dict)


def nfold_evaluate_mlp_single_mutation(saved_dir, data_dir, file_name, task, device):
    """
    evaluate n-fold saved models on the given task
    """
    result_dict = load_data(saved_dir + "mlp_result_dict.pkl")
    nfold = result_dict["args"]["nfold"]
    input_dim = len(AA2ID)*len(LACI_WT)
    hidden_dim = result_dict["args"]["hidden_dim"]
    
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

        mlp_model = MLP(input_dim, hidden_dim, 1)
        mlp_model.load_state_dict(torch.load(saved_dir + "mlp_fold{}_model.pt".format(n)))
        test_metric = evaluate_mlp_single_mutation(testing_dataset, mlp_model, 
                                                   result_dict["args"]["train_objective"], 
                                                   task, device)
        test_metric_list.append(test_metric)

    if task == "binary":
        print("avg binary AUC: {}".format(np.mean(test_metric_list)))
        print("std binary AUC: {}".format(np.std(test_metric_list)))
    elif task == "score":
        print("avg spearman corr: {}".format(np.mean(test_metric_list)))
        print("std spearman corr: {}".format(np.std(test_metric_list)))
    
    return test_metric_list


def nfold_evaluate_mlp_multiple_mutations(saved_dir, data_dir, score_computation, task, device):
    """
    evaluate n-fold saved models on the given task
    """
    result_dict = load_data(saved_dir + "mlp_result_dict.pkl")
    nfold = result_dict["args"]["nfold"]
    input_dim = len(AA2ID)*len(LACI_WT)
    hidden_dim = result_dict["args"]["hidden_dim"]
    
    test_metric_list = []
    print("loading data...")
    testing_data = load_data(data_dir)
    testing_dataset = LacIDatasetOneHot(testing_data, AA2ID, ID2AA)
    print("loading data done")

    for n in range(nfold):

        mlp_model = MLP(input_dim, hidden_dim, 1)
        mlp_model.load_state_dict(torch.load(saved_dir + "mlp_fold{}_model.pt".format(n)))
        test_metric = evaluate_mlp_multiple_mutations(testing_dataset, mlp_model, score_computation,
                                                      result_dict["args"]["train_objective"], 
                                                      task, device)
        test_metric_list.append(test_metric)

    if task == "binary":
        print("avg binary AUC: {}".format(np.mean(test_metric_list)))
        print("std binary AUC: {}".format(np.std(test_metric_list)))
    elif task == "score":
        print("avg spearman corr: {}".format(np.mean(test_metric_list)))
        print("std spearman corr: {}".format(np.std(test_metric_list)))
    
    return test_metric_list

