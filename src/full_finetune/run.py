import numpy as np
import torch
import esm
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils import *
from src.full_finetune.data import *
from src.full_finetune.models import *
from src.full_finetune.utils import *
from src.constants import *
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import BinaryAUROC
from torchmetrics.regression import SpearmanCorrCoef


def finetune_ESM2_attention_head(training_dataset, validation_dataset, pretrained_model, batch_converter,
                                   max_epoch, batch_size, learning_rate, early_stop=5, device="cpu"):

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                                     collate_fn=collate_function_sequence_dataset, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, 
                                       collate_fn=collate_function_sequence_dataset, drop_last=True)
 
    model = ESM2AttentionHead(pretrained_model, 1)
    model.to(device)
    loss_fn = torch.nn.MSELoss()

    if isinstance(learning_rate, list):
        # learning rate schedule should be provided as [peak_learning_rate, warmup_step, decay_step]
        print("using learning rate scheduler")
        lr_lambda_fn = generate_lr_schedule_function(learning_rate[1], learning_rate[2])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[0])
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_fn)
        scheduling = True
    else:
        # use fixed learning rate
        print("using a fixed learning rate")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduling = False

    print("training starts...")
    training_loss_record = []
    validation_loss_record = []
    validation_metric_record = []
    best_metric = 0
    best_model = model.state_dict()
    best_epoch = 0

    batch_loss_sum = 0.
    num_total_batch = len(validation_dataloader)
    model.eval()
    for mutation_sequence_pair_batch, label_batch in tqdm(validation_dataloader):
        
        with torch.no_grad():
            mutation_batch, sequence_batch, x_batch = batch_converter(mutation_sequence_pair_batch)
            label_batch = label_batch.to(dtype=torch.float32).to(device)
            x_batch = x_batch.to(device)
            logits = model(x_batch) 
            label_batch = label_batch.view(logits.size()) 
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

        model.train()
        for mutation_sequence_pair_batch, label_batch in tqdm(training_dataloader):
    
            optimizer.zero_grad()
            mutation_batch, sequence_batch, x_batch = batch_converter(mutation_sequence_pair_batch)
            label_batch = label_batch.to(dtype=torch.float32).to(device)
            x_batch = x_batch.to(device)
            logits = model(x_batch) 
            label_batch = label_batch.view(logits.size()) 
            loss_batch = loss_fn(logits, label_batch)
            loss_batch.backward()
            optimizer.step()
            if scheduling:
                scheduler.step()

            batch_loss_sum += loss_batch.item()

        training_loss = batch_loss_sum/num_total_batch
        training_loss_record.append(training_loss)

        print("calculating validation loss...")
        batch_loss_sum = 0.
        num_total_batch = len(validation_dataloader)

        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()
        
        model.eval()
        for mutation_sequence_pair_batch, label_batch in tqdm(validation_dataloader):
        
            with torch.no_grad():
                mutation_batch, sequence_batch, x_batch = batch_converter(mutation_sequence_pair_batch)
                label_batch = label_batch.to(dtype=torch.float32).to(device)
                x_batch = x_batch.to(device)
                logits = model(x_batch) 
                label_batch = label_batch.view(logits.size()) 
                loss_batch = loss_fn(logits, label_batch)
                batch_loss_sum += loss_batch.item()
                CRITERION.update(logits, label_batch)

        validation_loss = batch_loss_sum/num_total_batch
        validation_loss_record.append(validation_loss)
        validation_metric = CRITERION.compute().item()
        validation_metric_record.append(validation_metric)
        
        if validation_metric > best_metric:
            best_metric = validation_metric
            best_epoch = e+1
            best_model = model.state_dict()

        print("training loss: {}".format(training_loss))
        print("validation loss: {}".format(validation_loss))
        print("validation spearman corr: {}".format(validation_metric))

    print("best model at epoch {}".format(best_epoch))
    print("saving results...")
    result_dict = {"training_epoch" : e+1, "batch_size" : batch_size, "learning_rate" : learning_rate,
                   "training_loss_record" : training_loss_record, "validation_loss_record" : validation_loss_record,
                   "validation_metric_record" : validation_metric_record}

    model.load_state_dict(best_model)

    return model, result_dict


def finetune_ESM2_meanpooling_head(training_dataset, validation_dataset, pretrained_model, batch_converter,
                                   max_epoch, batch_size, learning_rate, early_stop=5, device="cpu"):

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                                     collate_fn=collate_function_sequence_dataset, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, 
                                       collate_fn=collate_function_sequence_dataset, drop_last=True)
 
    model = ESM2MeanpoolingHead(pretrained_model, 1)
    model.to(device)
    loss_fn = torch.nn.MSELoss()

    if isinstance(learning_rate, list):
        # learning rate schedule should be provided as [peak_learning_rate, warmup_step, decay_step]
        print("using learning rate scheduler")
        lr_lambda_fn = generate_lr_schedule_function(learning_rate[1], learning_rate[2])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[0])
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_fn)
        scheduling = True
    else:
        # use fixed learning rate
        print("using a fixed learning rate")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduling = False

    print("training starts...")
    training_loss_record = []
    validation_loss_record = []
    validation_metric_record = []
    best_metric = 0
    best_model = model.state_dict()
    best_epoch = 0

    batch_loss_sum = 0.
    num_total_batch = len(validation_dataloader)
    model.eval()
    for mutation_sequence_pair_batch, label_batch in tqdm(validation_dataloader):
        
        with torch.no_grad():
            mutation_batch, sequence_batch, x_batch = batch_converter(mutation_sequence_pair_batch)
            label_batch = label_batch.to(dtype=torch.float32).to(device)
            x_batch = x_batch.to(device)
            logits = model(x_batch) 
            label_batch = label_batch.view(logits.size()) 
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

        model.train()
        for mutation_sequence_pair_batch, label_batch in tqdm(training_dataloader):
    
            optimizer.zero_grad()
            mutation_batch, sequence_batch, x_batch = batch_converter(mutation_sequence_pair_batch)
            label_batch = label_batch.to(dtype=torch.float32).to(device)
            x_batch = x_batch.to(device)
            logits = model(x_batch) 
            label_batch = label_batch.view(logits.size()) 
            loss_batch = loss_fn(logits, label_batch)
            loss_batch.backward()
            optimizer.step()
            if scheduling:
                scheduler.step()
            batch_loss_sum += loss_batch.item()

        training_loss = batch_loss_sum/num_total_batch
        training_loss_record.append(training_loss)

        print("calculating validation loss...")
        batch_loss_sum = 0.
        num_total_batch = len(validation_dataloader)

        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()
        
        model.eval()
        for mutation_sequence_pair_batch, label_batch in tqdm(validation_dataloader):
        
            with torch.no_grad():
                mutation_batch, sequence_batch, x_batch = batch_converter(mutation_sequence_pair_batch)
                label_batch = label_batch.to(dtype=torch.float32).to(device)
                x_batch = x_batch.to(device)
                logits = model(x_batch) 
                label_batch = label_batch.view(logits.size()) 
                loss_batch = loss_fn(logits, label_batch)
                batch_loss_sum += loss_batch.item()
                CRITERION.update(logits, label_batch)


        validation_loss = batch_loss_sum/num_total_batch
        validation_loss_record.append(validation_loss)
        validation_metric = CRITERION.compute().item()
        validation_metric_record.append(validation_metric)
        
        if validation_metric > best_metric:
            best_metric = validation_metric
            best_epoch = e+1
            best_model = model.state_dict()

        print("training loss: {}".format(training_loss))
        print("validation loss: {}".format(validation_loss))
        print("validation spearman corr: {}".format(validation_metric))

    print("best model at epoch {}".format(best_epoch))
    print("saving results...")
    result_dict = {"training_epoch" : e+1, "batch_size" : batch_size, "learning_rate" : learning_rate,
                   "training_loss_record" : training_loss_record, "validation_loss_record" : validation_loss_record,
                   "validation_metric_record" : validation_metric_record}

    model.load_state_dict(best_model)

    return model, result_dict


def finetune_ESM2_position_head(training_dataset, validation_dataset, pretrained_model, batch_converter,
                                   max_epoch, batch_size, learning_rate, early_stop=5, device="cpu"):

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                                     collate_fn=collate_function_sequence_dataset, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, 
                                       collate_fn=collate_function_sequence_dataset, drop_last=True)
 
    model = ESM2MutationPositionHead(pretrained_model, 1)
    model.to(device)
    loss_fn = torch.nn.MSELoss()

    if isinstance(learning_rate, list):
        # learning rate schedule should be provided as [peak_learning_rate, warmup_step, decay_step]
        print("using learning rate scheduler")
        lr_lambda_fn = generate_lr_schedule_function(learning_rate[1], learning_rate[2])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate[0])
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda_fn)
        scheduling = True
    else:
        # use fixed learning rate
        print("using a fixed learning rate")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduling = False

    print("training starts...")
    training_loss_record = []
    validation_loss_record = []
    validation_metric_record = []
    best_metric = 0
    best_model = model.state_dict()
    best_epoch = 0

    batch_loss_sum = 0.
    num_total_batch = len(validation_dataloader)
    model.eval()
    for mutation_sequence_pair_batch, label_batch in tqdm(validation_dataloader):
        
        with torch.no_grad():
            mutation_batch, sequence_batch, x_batch = batch_converter(mutation_sequence_pair_batch)
            mutation_position_batch = get_mutation_position(mutation_batch, adjust_position=1)
            mutation_position_batch = mutation_position_batch.to(device)
            label_batch = label_batch.to(dtype=torch.float32).to(device)
            x_batch = x_batch.to(device)
            logits = model(x_batch, mutation_position_batch) 
            label_batch = label_batch.view(logits.size()) 
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

        model.train()
        for mutation_sequence_pair_batch, label_batch in tqdm(training_dataloader):
    
            optimizer.zero_grad()
            mutation_batch, sequence_batch, x_batch = batch_converter(mutation_sequence_pair_batch)
            mutation_position_batch = get_mutation_position(mutation_batch, adjust_position=1)
            mutation_position_batch = mutation_position_batch.to(device)
            label_batch = label_batch.to(dtype=torch.float32).to(device)
            x_batch = x_batch.to(device)
            logits = model(x_batch, mutation_position_batch) 
            label_batch = label_batch.view(logits.size()) 
            loss_batch = loss_fn(logits, label_batch)
            loss_batch.backward()
            optimizer.step()
            if scheduling:
                scheduler.step()
            batch_loss_sum += loss_batch.item()
        
        training_loss = batch_loss_sum/num_total_batch
        training_loss_record.append(training_loss)

        print("calculating validation loss...")
        batch_loss_sum = 0.
        num_total_batch = len(validation_dataloader)

        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()
        
        model.eval()
        for mutation_sequence_pair_batch, label_batch in tqdm(validation_dataloader):
        
            with torch.no_grad():
                mutation_batch, sequence_batch, x_batch = batch_converter(mutation_sequence_pair_batch)
                mutation_position_batch = get_mutation_position(mutation_batch, adjust_position=1)
                mutation_position_batch = mutation_position_batch.to(device)
                label_batch = label_batch.to(dtype=torch.float32).to(device)
                x_batch = x_batch.to(device)
                logits = model(x_batch, mutation_position_batch) 
                label_batch = label_batch.view(logits.size()) 
                loss_batch = loss_fn(logits, label_batch)
                batch_loss_sum += loss_batch.item()
                CRITERION.update(logits, label_batch)

        validation_loss = batch_loss_sum/num_total_batch
        validation_loss_record.append(validation_loss)
        validation_metric = CRITERION.compute().item()
        validation_metric_record.append(validation_metric)
        
        if validation_metric > best_metric:
            best_metric = validation_metric
            best_epoch = e+1
            best_model = model.state_dict()

        print("training loss: {}".format(training_loss))
        print("validation loss: {}".format(validation_loss))
        print("validation spearman corr: {}".format(validation_metric))

    print("best model at epoch {}".format(best_epoch))
    print("saving results...")
    result_dict = {"training_epoch" : e+1, "batch_size" : batch_size, "learning_rate" : learning_rate,
                   "training_loss_record" : training_loss_record, "validation_loss_record" : validation_loss_record,
                   "validation_metric_record" : validation_metric_record}

    model.load_state_dict(best_model)

    return model, result_dict


def evaluate_full_finetuned_model_single_mutation(testing_dataset, model, head, batch_converter, task, device):

    testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=False, 
                                    collate_fn=collate_function_sequence_dataset, drop_last=False)
        
    model.to(device)

    if task == "binary":
        CRITERION = BinaryAUROC()
        CRITERION.reset()
    elif task == "score":
        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()

    model.eval()
    for mutation_sequence_pair_batch, label_batch in tqdm(testing_dataloader):

        with torch.no_grad():
            mutation_batch, sequence_batch, x_batch = batch_converter(mutation_sequence_pair_batch)
            label_batch = label_batch.to(dtype=torch.float32).to(device)
            x_batch = x_batch.to(device)

            if head in ["mean", "attention"]:
                mutation_scores = model(x_batch) 
            elif head == "position":
                mutation_position_batch = get_mutation_position(mutation_batch, adjust_position=1)
                mutation_position_batch = mutation_position_batch.to(device)
                mutation_scores = model(x_batch, mutation_position_batch) 

            CRITERION.update(mutation_scores.to(device), label_batch.view(mutation_scores.size()))

    return CRITERION.compute().item()


def evaluate_full_finetuned_model_multiple_mutations(testing_dataset, model, head, batch_converter, 
                                                     score_computation, task, device):

    testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=False, 
                                    collate_fn=collate_function_sequence_dataset, drop_last=False)
        
    model.to(device)

    if task == "binary":
        CRITERION = BinaryAUROC()
        CRITERION.reset()
    elif task == "score":
        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()

    model.eval()
    for mutation_sequence_pair_batch, label_batch in tqdm(testing_dataloader):

        with torch.no_grad():
            mutation_batch, sequence_batch, x_batch = batch_converter(mutation_sequence_pair_batch)
            label_batch = label_batch.to(dtype=torch.float32).to(device)
            x_batch = x_batch.to(device)

            if head in ["mean", "attention"]:

                if score_computation == "general":
                    mutation_scores = model(x_batch) 
                elif score_computation == "additive":
                    mutation_scores = compute_additive_mutation_score(model, x_batch, mutation_batch, batch_converter)

            elif head == "position":

                if score_computation == "general":
                    mutation_position_batch = get_mutation_position(mutation_batch, adjust_position=1) # (batch_size, num_mutations)
                    mutation_position_batch = mutation_position_batch.to(device)
                    mutation_scores = model(x_batch, mutation_position_batch) 
                elif score_computation == "additive":
                    mutation_scores = compute_additive_mutation_score_position_pooling(model, x_batch, mutation_batch, batch_converter) 

        CRITERION.update(mutation_scores.to(device), label_batch.view(mutation_scores.size()))

    return CRITERION.compute().item()


def nfold_full_finetune(data_dir, saving_dir, file_name, pretrained_model, head, nfold, max_epoch, batch_size, 
                              learning_rate, early_stop, device):

    nfold_result_dict = dict()
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold), 2)

    for n in range(nfold):
        
        print("loading data...")
        validation_idx = np.array(nfold_idx[n])
        testing_idx = np.array(nfold_idx[n+1])
        training_idx = np.array(nfold_idx[n+2:n+nfold])

        training_data = merge_folds(data_nfold[training_idx])
        validation_data = data_nfold[validation_idx]

        training_dataset = SequenceDataset(training_data)
        validation_dataset = SequenceDataset(validation_data)
        print("loading data done")

        print("loading pre-trained ESM2...")
        if pretrained_model == "esm2_t6_8M_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            batch_converter = alphabet.get_batch_converter()     
        elif pretrained_model == "esm2_t12_35M_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            batch_converter = alphabet.get_batch_converter()
        elif pretrained_model == "esm2_t30_150M_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            batch_converter = alphabet.get_batch_converter()
        elif pretrained_model == "esm2_t33_650M_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            batch_converter = alphabet.get_batch_converter()
        else:
            raise ValueError("wrong pretrained model input")
        print("loading pre-trained ESM2 done")

        if head == "attention":
            best_model, result_dict = finetune_ESM2_attention_head(training_dataset, validation_dataset, ESM2_model, 
                                                                batch_converter, max_epoch, batch_size, learning_rate, 
                                                                early_stop, device)
        elif head == "mean":
            best_model, result_dict = finetune_ESM2_meanpooling_head(training_dataset, validation_dataset, ESM2_model, 
                                                                     batch_converter,
                                                            max_epoch, batch_size, learning_rate, 
                                                            early_stop, device)
        elif head == "position":
            best_model, result_dict = finetune_ESM2_position_head(training_dataset, validation_dataset, ESM2_model, batch_converter,
                                                             max_epoch, batch_size, learning_rate, 
                                                             early_stop, device)   
        else:
            raise ValueError("wrong head value assigned")

        nfold_result_dict[n] = copy.deepcopy(result_dict)
        torch.save(best_model.state_dict(), saving_dir + head + "_fold{}_model.pt".format(n))

    save_data(saving_dir + head + "_result_dict.pkl", nfold_result_dict)


def nfold_evaluate_full_finetune_single_mutation(saved_dir, data_dir, file_name, pretrained_model, 
                                                 head, nfold, task, device):
    """
    evaluate n-fold saved models on the given task
    """

    test_metric_list = []
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)

    for n in range(nfold):
        
        print("loading data...")
        validation_idx = np.array(nfold_idx[n])
        testing_idx = np.array(nfold_idx[n+1])
        training_idx = np.array(nfold_idx[n+2:n+nfold])

        testing_data = data_nfold[testing_idx]
        testing_dataset = SequenceDataset(testing_data)
        print("loading data done")

        print("loading pre-trained ESM2...")
        if pretrained_model == "esm2_t6_8M_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            batch_converter = alphabet.get_batch_converter()    
        elif pretrained_model == "esm2_t12_35M_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            batch_converter = alphabet.get_batch_converter()
        elif pretrained_model == "esm2_t30_150M_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            batch_converter = alphabet.get_batch_converter()
        elif pretrained_model == "esm2_t33_650M_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            batch_converter = alphabet.get_batch_converter()
        else:
            raise ValueError("wrong pretrained model input")
        print("loading pre-trained ESM2 done")

        if head == "attention":
            model = ESM2AttentionHead(ESM2_model, 1)
            model.load_state_dict(torch.load(saved_dir + head + "_fold{}_model.pt".format(n)))

        elif head == "mean":
            model = ESM2MeanpoolingHead(ESM2_model, 1)
            model.load_state_dict(torch.load(saved_dir + head + "_fold{}_model.pt".format(n)))

        elif head == "position":
            model = ESM2MutationPositionHead(ESM2_model, 1)
            model.load_state_dict(torch.load(saved_dir + head + "_fold{}_model.pt".format(n)))

        test_metric = evaluate_full_finetuned_model_single_mutation(testing_dataset, model, head, 
                                                    batch_converter, task, device)
        test_metric_list.append(test_metric)

    if task == "binary":
        print("avg binary AUC: {}".format(np.mean(test_metric_list)))
        print("std binary AUC: {}".format(np.std(test_metric_list)))
    elif task == "score":
        print("avg spearman corr: {}".format(np.mean(test_metric_list)))
        print("std spearman corr: {}".format(np.std(test_metric_list)))

    print(test_metric_list)
    
    return test_metric_list


def nfold_evaluate_full_finetune_multiple_mutations(saved_dir, data_dir, pretrained_model, head, nfold, task, 
                                                    score_computation, device):
    """
    evaluate n-fold saved confit models on the given task
    """

    test_metric_list = []
    testing_data = load_data(data_dir)
    testing_dataset = SequenceDataset(testing_data)

    for n in range(nfold):

        print("loading pre-trained ESM2...")
        if pretrained_model == "esm2_t6_8M_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
            batch_converter = alphabet.get_batch_converter()    
        elif pretrained_model == "esm2_t12_35M_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
            batch_converter = alphabet.get_batch_converter()
        elif pretrained_model == "esm2_t30_150M_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
            batch_converter = alphabet.get_batch_converter()
        elif pretrained_model == "esm2_t33_650M_UR50D":
            ESM2_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            batch_converter = alphabet.get_batch_converter()
        else:
            raise ValueError("wrong pretrained model input")
        print("loading pre-trained ESM2 done")

        if head == "attention":
            model = ESM2AttentionHead(ESM2_model, 1)
            model.load_state_dict(torch.load(saved_dir + head + "_fold{}_model.pt".format(n)))

        elif head == "mean":
            model = ESM2MeanpoolingHead(ESM2_model, 1)
            model.load_state_dict(torch.load(saved_dir + head + "_fold{}_model.pt".format(n)))

        elif head == "position":
            model = ESM2MutationPositionHead(ESM2_model, 1)
            model.load_state_dict(torch.load(saved_dir + head + "_fold{}_model.pt".format(n)))

        if score_computation == "general":
            test_metric = evaluate_full_finetuned_model_multiple_mutations(testing_dataset, 
                                                                        model, 
                                                                        head,
                                                                        batch_converter,
                                                                        "general",
                                                                        task, device)

        elif score_computation == "additive":
            test_metric = evaluate_full_finetuned_model_multiple_mutations(testing_dataset, 
                                                                        model, 
                                                                        head,
                                                                        batch_converter,
                                                                        "additive",
                                                                        task, device)
        test_metric_list.append(test_metric)

    if task == "binary":
        print("avg binary AUC: {}".format(np.mean(test_metric_list)))
        print("std binary AUC: {}".format(np.std(test_metric_list)))
    elif task == "score":
        print("avg spearman corr: {}".format(np.mean(test_metric_list)))
        print("std spearman corr: {}".format(np.std(test_metric_list)))

    print(test_metric_list)
    
    return test_metric_list