import numpy as np
import torch
from esm.models.esmc import ESMC
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.utils import *
from src.esmc_confit.data import *
from src.esmc_confit.loss import *
from src.constants import *
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.classification import BinaryAUROC
from torchmetrics.regression import SpearmanCorrCoef


def finetune_confit(training_dataset, validation_dataset, model,
                    max_epoch, batch_size, learning_rate, lambda_reg, early_stop, device="cpu"):
    
    model_reg = copy.deepcopy(model)
    model_reg.load_state_dict(model.state_dict())
    model_reg.to(device)
    model_reg.eval() # model_reg is frozen
    model.to(device)

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                                     collate_fn=collate_function_sequence_dataset, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, 
                                       collate_fn=collate_function_sequence_dataset, drop_last=True)
    
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
    training_bt_loss_record = []
    training_kl_loss_record = []
    validation_loss_record = []
    validation_bt_loss_record = []
    validation_kl_loss_record = []
    validation_metric_record = []
    best_metric = 0
    best_model = model.state_dict()
    best_epoch = 0

    batch_loss_sum = 0.
    loss_BT_sum = 0.
    loss_KL_sum = 0.
    num_total_batch = len(validation_dataloader)
    model.eval()
    for mutation_batch, sequence_batch, label_batch in tqdm(validation_dataloader):

        with torch.no_grad():
            x_batch = model.tokenizer(sequence_batch)["input_ids"]
            x_batch = torch.tensor(x_batch).to(device)
            wt_batch = torch.tensor(model.tokenizer([LACI_WT])["input_ids"]).repeat(batch_size, 1).to(device)
            label_batch = label_batch.to(device) # (batch_size)

            mutation_scores, logits = compute_mutation_score(model, x_batch, wt_batch, mutation_batch)
            loss_BT = compute_BT_loss(mutation_scores, label_batch)
            output_reg = model_reg(wt_batch)
            logits_reg = output_reg.sequence_logits
            loss_KL = compute_KL_loss(logits, logits_reg, wt_batch)

            loss_batch = loss_BT + lambda_reg*loss_KL
            loss_BT_sum += loss_BT.item()
            loss_KL_sum += loss_KL.item()
            batch_loss_sum += loss_batch.item()
        
    validation_loss = batch_loss_sum/num_total_batch
    validation_loss_BT = loss_BT_sum/num_total_batch
    validation_loss_KL = loss_KL_sum/num_total_batch
    validation_loss_record.append(validation_loss)
    validation_bt_loss_record.append(validation_loss_BT)
    validation_kl_loss_record.append(validation_loss_KL)
    print("initial validation loss: {}".format(validation_loss))
    print("initial validation BT loss: {}".format(validation_loss_BT))
    print("initial validation KL loss: {}".format(validation_loss_KL))

    for e in range(max_epoch):

        if early_stop != None:
            if (e+1-best_epoch) > early_stop:
                # if the loss did not decrease for pre-defined number of epoch in a row, stop training
                break

        print("epoch {}".format(e+1))
        batch_loss_sum = 0.
        loss_BT_sum = 0.
        loss_KL_sum = 0.
        num_total_batch = len(training_dataloader)
        model.train()
        for mutation_batch, sequence_batch, label_batch in tqdm(training_dataloader):

            x_batch = model.tokenizer(sequence_batch)["input_ids"]
            x_batch = torch.tensor(x_batch).to(device)
            wt_batch = torch.tensor(model.tokenizer([LACI_WT])["input_ids"]).repeat(batch_size, 1).to(device)
            label_batch = label_batch.to(device) # (batch_size)

            mutation_scores, logits = compute_mutation_score(model, x_batch, wt_batch, mutation_batch)
            loss_BT = compute_BT_loss(mutation_scores, label_batch)
            output_reg = model_reg(wt_batch)
            logits_reg = output_reg.sequence_logits
            loss_KL = compute_KL_loss(logits, logits_reg, wt_batch)

            loss_batch = loss_BT + lambda_reg*loss_KL

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            if scheduling:
                scheduler.step()

            loss_BT_sum += loss_BT.item()
            loss_KL_sum += loss_KL.item()
            batch_loss_sum += loss_batch.item()
        
        training_loss = batch_loss_sum/num_total_batch
        training_loss_BT = loss_BT_sum/num_total_batch
        training_loss_KL = loss_KL_sum/num_total_batch
        training_loss_record.append(training_loss)
        training_bt_loss_record.append(training_loss_BT)
        training_kl_loss_record.append(training_loss_KL)

        print("calculating validation loss...")
        batch_loss_sum = 0.
        loss_BT_sum = 0.
        loss_KL_sum = 0.
        num_total_batch = len(validation_dataloader)

        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()
        
        model.eval()
        for mutation_batch, sequence_batch, label_batch in tqdm(validation_dataloader):

            with torch.no_grad():
                x_batch = model.tokenizer(sequence_batch)["input_ids"]
                x_batch = torch.tensor(x_batch).to(device)
                wt_batch = torch.tensor(model.tokenizer([LACI_WT])["input_ids"]).repeat(batch_size, 1).to(device)
                label_batch = label_batch.to(device) # (batch_size)

                mutation_scores, logits = compute_mutation_score(model, x_batch, wt_batch, mutation_batch)
                loss_BT = compute_BT_loss(mutation_scores, label_batch)
                output_reg = model_reg(wt_batch)
                logits_reg = output_reg.sequence_logits
                loss_KL = compute_KL_loss(logits, logits_reg, wt_batch)

                loss_batch = loss_BT + lambda_reg*loss_KL
                loss_BT_sum += loss_BT.item()
                loss_KL_sum += loss_KL.item()
                batch_loss_sum += loss_batch.item()
                CRITERION.update(mutation_scores, label_batch.to(torch.float32))
        
        validation_metric = CRITERION.compute().item()
        validation_loss = batch_loss_sum/num_total_batch
        validation_loss_BT = loss_BT_sum/num_total_batch
        validation_loss_KL = loss_KL_sum/num_total_batch
        validation_loss_record.append(validation_loss)
        validation_bt_loss_record.append(validation_loss_BT)
        validation_kl_loss_record.append(validation_loss_KL)
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
                   "training_bt_loss_record" : training_bt_loss_record, "training_kl_loss_record" : training_kl_loss_record,
                   "validation_bt_loss_record" : validation_bt_loss_record, 
                   "validation_kl_loss_record" : validation_kl_loss_record,
                   "validation_metric_record" : validation_metric_record}

    model.load_state_dict(best_model)

    return model, result_dict


def nfold_finetune_confit(data_dir, saving_dir, file_name, pretrained_model, nfold, max_epoch, batch_size, 
                          learning_rate, lambda_reg, early_stop, device):

    nfold_result_dict = dict()
    data_nfold = np.array(load_nfold_data(data_dir, file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)

    for n in range(nfold):
        
        print("loading data...")
        validation_idx = np.array(nfold_idx[n])
        testing_idx = np.array(nfold_idx[n+1])
        training_idx = np.array(nfold_idx[n+2:n+nfold])

        training_data = merge_folds(data_nfold[training_idx])
        validation_data = data_nfold[validation_idx]
        testing_data = data_nfold[testing_idx]

        training_dataset = SequenceDataset(training_data)
        validation_dataset = SequenceDataset(validation_data)
        testing_dataset = SequenceDataset(testing_data)
        print("loading data done")

        print("loading pre-trained ESMC...")
        if pretrained_model == "300m":
            ESMC_model = ESMC.from_pretrained("esmc_300m")
        elif pretrained_model == "600m":
            ESMC_model = ESMC.from_pretrained("esmc_600m")
        else:
            raise ValueError("wrong pretrained model input")
        print("loading pre-trained ESM2 done")

        best_model, result_dict = finetune_confit(training_dataset, validation_dataset, ESMC_model,
                                                  max_epoch, batch_size, learning_rate, lambda_reg, 
                                                  early_stop, device)

        nfold_result_dict[n] = copy.deepcopy(result_dict)
        torch.save(best_model.state_dict(), saving_dir+"ESMC_confit_fold{}_model.pt".format(n))

    save_data(saving_dir+"ESMC_confit_result_dict.pkl", nfold_result_dict)