import torch
import copy
import numpy as np
from src.utils import *
from src.full_finetuning.data import *
from tape import TAPETokenizer
from tape.models.modeling_unirep import *
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC
from torchmetrics.regression import SpearmanCorrCoef

class SimpleMLP(torch.nn.Module):
    """
    simple MLP with one hidden layer
    """

    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout, inplace=True),
            torch.nn.Linear(input_dim, output_dim))
        
    def forward(self, x):

        return self.main(x)
        
class UniRep_MeanPooling(torch.nn.Module):
    # UniRep with MLP head on mean pooled sequence

    def __init__(self, unirep_model, output_dim, dropout):
        super(UniRep_MeanPooling, self).__init__()

        self.unirep_model = unirep_model
        self.mlp = SimpleMLP(unirep_model.config.hidden_size, output_dim, dropout)
        
    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.unirep_model(input_ids, input_mask=input_mask)
        sequence_output, last_output = outputs
        # mean pooling over the entire sequence
        # the first and last tokens are <CLS> and <EOS>
        pooled_output = sequence_output[:,:,:].mean(1) # batch_size * embedding_dim

        return self.mlp(pooled_output)
    
class UniRep_FinalHidden(torch.nn.Module):
    # UniRep with MLP head on the concat of final hidden and cell state

    def __init__(self, unirep_model, output_dim, dropout):
        super(UniRep_FinalHidden, self).__init__()

        self.unirep_model = unirep_model
        self.mlp = SimpleMLP(unirep_model.config.hidden_size*2, output_dim, dropout)
        
    def forward(self, input_ids, input_mask=None, targets=None):

        outputs = self.unirep_model(input_ids, input_mask=input_mask)
        sequence_output, last_output = outputs
        # last output: batch_size * 2xhidden_dim

        return self.mlp(last_output)
    
def collate_function(batch):
    
    mutation_sequence_pair, label = list(zip(*batch))
    
    return mutation_sequence_pair, torch.Tensor(label)

def batch_convert(tokenizer, mutation_sequence_pair_batch):

    encoded_sequence_list = []

    for mut, seq in mutation_sequence_pair_batch:
        encoded_sequence_list.append(tokenizer.encode(seq))

    return torch.Tensor(encoded_sequence_list).long()

def finetune_UniRep(training_dataset, validation_dataset, pretrained_model, model, tokenizer,
                           max_epoch, batch_size, learning_rate, dropout, task, early_stop=5, 
                           freeze_unirep=True, device="cpu"):

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function, drop_last=True)
 
    print("building and initializing model...")
    if model == "meanpooling":
        unirep_finetune_model = UniRep_MeanPooling(pretrained_model, 1, dropout)
        unirep_finetune_model.to(device)
    elif model == "finalhidden":
        unirep_finetune_model = UniRep_FinalHidden(pretrained_model, 1, dropout)
        unirep_finetune_model.to(device)
    else:
        raise ValueError("wrong model assigned")

    if task == "binary":
        Sigmoid = torch.nn.Sigmoid()
        loss_fn = torch.nn.BCELoss()
    elif task == "score":
        loss_fn = torch.nn.MSELoss()
    else:
        raise ValueError("wrong task assigned")
    
    Optimizer = torch.optim.Adam(unirep_finetune_model.parameters(), lr=learning_rate)
    print("building and initializing model done...")

    if freeze_unirep:
        print("freezing UniRep parameters..")
        for param in unirep_finetune_model.unirep_model.parameters():
            param.requires_grad = False

    print("training starts...")
    training_loss_per_epoch = []
    validation_loss_per_epoch = []
    validation_metric_per_epoch = []
    best_metric = 0
    best_model = None
    best_epoch = 0
    """
    batch_loss_sum = 0.
    num_total_batch = len(validation_dataloader)
    unirep_finetune_model.eval()
    for mutation_sequence_pair_batch, label_batch in tqdm(validation_dataloader):
        
        with torch.no_grad():
            x_batch = batch_convert(tokenizer, mutation_sequence_pair_batch)
            label_batch = label_batch.to(device)
            x_batch = x_batch.to(device)
            logits = unirep_finetune_model(x_batch) # batch_size * num_class
            label_batch = label_batch.view(logits.size()) # batch_size * num_class

            if task == "binary":
                loss_batch = loss_fn(Sigmoid(logits), label_batch)
            elif task == "score":
                loss_batch = loss_fn(logits, label_batch)
            
            batch_loss_sum += loss_batch.item()
        
    validation_loss = batch_loss_sum/num_total_batch
    print("initial validation loss: {}".format(validation_loss))
    """
    for e in range(max_epoch):

        if early_stop != None:
            if (e+1-best_epoch) > early_stop:
                # if the loss did not decrease for pre-defined number of epoch in a row, stop training
                break

        print("epoch {}".format(e+1))
        batch_loss_sum = 0.
        num_total_batch = len(training_dataloader)

        unirep_finetune_model.train()
        for mutation_sequence_pair_batch, label_batch in tqdm(training_dataloader):
            
            Optimizer.zero_grad()
            x_batch = batch_convert(tokenizer, mutation_sequence_pair_batch)
            label_batch = label_batch.to(device)
            x_batch = x_batch.to(device)
            logits = unirep_finetune_model(x_batch) # batch_size * num_class
            label_batch = label_batch.view(logits.size()) # batch_size * num_class

            if task == "binary":
                loss_batch = loss_fn(Sigmoid(logits), label_batch)
            elif task == "score":
                loss_batch = loss_fn(logits, label_batch)

            loss_batch.backward()
            Optimizer.step()
            batch_loss_sum += loss_batch.item()
        
        training_loss = batch_loss_sum/num_total_batch
        training_loss_per_epoch.append(training_loss)

        print("calculating validation loss...")
        batch_loss_sum = 0.
        num_total_batch = len(validation_dataloader)

        if task == "binary":
            CRITERION = BinaryAUROC()
            CRITERION.reset()
        elif task == "score":
            # no activation
            CRITERION = SpearmanCorrCoef()
            CRITERION.reset()
        
        unirep_finetune_model.eval()
        for mutation_sequence_pair_batch, label_batch in tqdm(validation_dataloader):

            with torch.no_grad():
                x_batch = batch_convert(tokenizer, mutation_sequence_pair_batch)
                label_batch = label_batch.to(device)
                x_batch = x_batch.to(device)
                logits = unirep_finetune_model(x_batch) # batch_size * num_class
                label_batch = label_batch.view(logits.size()) # batch_size * num_class

                if task == "binary":
                    loss_batch = loss_fn(Sigmoid(logits), label_batch)
                    CRITERION.update(Sigmoid(logits), label_batch)
                elif task == "score":
                    loss_batch = loss_fn(logits, label_batch)
                    CRITERION.update(logits, label_batch)

                batch_loss_sum += loss_batch.item()
        
        validation_loss = batch_loss_sum/num_total_batch
        validation_loss_per_epoch.append(validation_loss)
        validation_metric = CRITERION.compute().item()
        validation_metric_per_epoch.append(validation_metric)
        
        if validation_metric > best_metric:
            best_metric = validation_metric
            best_epoch = e+1
            best_model = unirep_finetune_model.state_dict()

        print("training loss: {}".format(training_loss))
        print("validation loss: {}".format(validation_loss))

        if task == "binary":
            print("validation AUC: {}".format(validation_metric))
        elif task == "score":
            print("validation spearman corr: {}".format(validation_metric))

    print("best model at epoch {}".format(best_epoch))
    print("saving results...")
    result_dict = {"training_epoch" : e+1, "batch_size" : batch_size, "learning_rate" : learning_rate,
                   "training_loss_per_epoch" : training_loss_per_epoch, "validation_loss_per_epoch" : validation_loss_per_epoch,
                   "validation_metric_per_epoch" : validation_metric_per_epoch}

    unirep_finetune_model.load_state_dict(best_model)

    return unirep_finetune_model, result_dict

def evaluate_UniRep(testing_dataset, unirep_model, tokenizer, task, device):

    testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=True, collate_fn=collate_function, drop_last=True)

    if task == "binary":
        Sigmoid = torch.nn.Sigmoid()
        CRITERION = BinaryAUROC()
        CRITERION.reset()
    elif task == "score":
        # no activation
        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()

    unirep_model.eval()
    for mutation_sequence_pair_batch, label_batch in tqdm(testing_dataloader):

        with torch.no_grad():
            x_batch = batch_convert(tokenizer, mutation_sequence_pair_batch)
            label_batch = label_batch.to(device)
            x_batch = x_batch.to(device)
            logits = unirep_model(x_batch) # batch_size * num_class

            if task == "binary":
                pred = Sigmoid(logits)
            elif task == "score":
                pred = copy.deepcopy(logits)

            label_batch = label_batch.view(logits.size())
            CRITERION.update(pred, label_batch)

    return CRITERION.compute().item()
    
def nfold_finetune_UniRep(data_dir, saving_dir, file_name, pretrained_model, model, nfold, max_epoch, batch_size, 
                        learning_rate, dropout, task, early_stop, freeze_unirep, device):

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

        training_dataset = LacIDataset(training_data)
        validation_dataset = LacIDataset(validation_data)
        testing_dataset = LacIDataset(testing_data)
        print("loading data done")

        print("loading pre-trained UniRep...")
        if pretrained_model == "unirep-1900":
            unirep_model = UniRepModel(UniRepConfig())
            unirep_model.from_pretrained("babbler-1900")
            tokenizer = TAPETokenizer(vocab='unirep')
        else:
            raise ValueError("wrong pretrained model input")
        print("loading pre-trained UniRep done")

        best_model, result_dict = finetune_UniRep(training_dataset, validation_dataset, unirep_model, model,
                                                tokenizer, max_epoch, batch_size, learning_rate, dropout,
                                                task, early_stop, freeze_unirep, device)
        
        test_metric = evaluate_UniRep(testing_dataset, best_model, tokenizer, task, device)
        
        if task == "binary":
            result_dict["test_AUC"] = test_metric
            print("test AUC: {}".format(test_metric))
        elif task == "score":
            result_dict["test_spearman_corr"] = test_metric
            print("test spearman corr: {}".format(test_metric))

        nfold_result_dict[n] = copy.deepcopy(result_dict)
        
        torch.save(best_model.state_dict(), saving_dir+"UniRep_finetuned_fold{}_model.pt".format(n))

    save_data(saving_dir+"UniRep_finetuned_result_dict.pkl", nfold_result_dict)