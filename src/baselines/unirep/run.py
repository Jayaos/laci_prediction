from tqdm import tqdm
from src.utils import *
from .model import SimpleLinear
from .data import collate_function_representation_dataset, RepresentatonDataset
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC
from torchmetrics.regression import SpearmanCorrCoef
from tape import TAPETokenizer
from tape.models.modeling_unirep import *


def train_unirep_head(training_dataset, validation_dataset, input_dim, max_epoch, batch_size, learning_rate, 
                      early_stop=5, device="cpu"):

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, 
                                     collate_fn=collate_function_representation_dataset, drop_last=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, 
                                       collate_fn=collate_function_representation_dataset, drop_last=True)
 
    model = SimpleLinear(input_dim, 1)
    model.to(device)
    loss_fn = torch.nn.MSELoss()
    Optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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
    for wt_residue, position, mt_residue, sequence_batch, representation_batch, label_batch in tqdm(validation_dataloader):
        
        with torch.no_grad():
            representation_batch = representation_batch.to(device)
            label_batch = label_batch.to(device)
            logits = model(representation_batch) # (batch_size, 1)
            label_batch = label_batch.view(logits.size()) + 2 # (batch_size, 1)
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
        for wt_residue, position, mt_residue, sequence_batch, representation_batch, label_batch in tqdm(training_dataloader):
            
            Optimizer.zero_grad()
            representation_batch = representation_batch.to(device)
            label_batch = label_batch.to(device)
            logits = model(representation_batch) # (batch_size, 1)
            label_batch = label_batch.view(logits.size()) # (batch_size, 1)
            loss_batch = loss_fn(logits, label_batch)
            loss_batch.backward()
            Optimizer.step()
            batch_loss_sum += loss_batch.item()
        
        training_loss = batch_loss_sum/num_total_batch
        training_loss_record.append(training_loss)

        print("calculating validation loss...")
        batch_loss_sum = 0.
        num_total_batch = len(validation_dataloader)

        CRITERION = SpearmanCorrCoef()
        CRITERION.reset()
        
        model.eval()
        for wt_residue, position, mt_residue, sequence_batch, representation_batch, label_batch in tqdm(validation_dataloader):

            with torch.no_grad():
                representation_batch = representation_batch.to(device)
                label_batch = label_batch.to(device)
                logits = model(representation_batch) # (batch_size, 1)
                label_batch = label_batch.view(logits.size()) + 2 # (batch_size, 1)
                loss_batch = loss_fn(logits, label_batch)
                print("logits : {}".format(logits.flatten()))
                print("label : {}".format(label_batch.flatten()))
                CRITERION.update(logits.flatten(), label_batch.flatten())
                batch_loss_sum += loss_batch.item()
        
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
                   "training_loss_record" : training_loss_record, 
                   "validation_loss_record" : validation_loss_record,
                   "validation_metric_record" : validation_metric_record}

    model.load_state_dict(best_model)

    return model, result_dict


def evaluate_unirep_head_single_mutation():

    ...


def evaluate_unirep_head_multiple_mutations():

    ...


def nfold_train_unirep_head(mutation_data_dir, representation_data_dir, saving_dir, mutation_file_name, 
                            pretrained_model, head, nfold, max_epoch, batch_size, learning_rate, early_stop, device):
    """
    train a prediction head on unirep representation
    this is same task with head_finetune on ESM representations
    """

    nfold_result_dict = dict()
    representation_data_dict = load_data(representation_data_dir + pretrained_model + "_single_mutation_representaton_data_dict.pkl")
    data_nfold = np.array(load_nfold_data(mutation_data_dir, mutation_file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)

    if pretrained_model == "unirep-1900":
        embed_dim = 1900

    for n in range(nfold):
        
        print("loading data...")
        validation_idx = np.array(nfold_idx[n])
        testing_idx = np.array(nfold_idx[n+1])
        training_idx = np.array(nfold_idx[n+2:n+nfold])

        training_data = merge_folds(data_nfold[training_idx])
        validation_data = data_nfold[validation_idx]

        training_dataset = RepresentatonDataset(training_data, representation_data_dict)
        validation_dataset = RepresentatonDataset(validation_data, representation_data_dict)
        print("loading data done")

        if head == "linear":
            # simple linear head
            best_model, result_dict = train_unirep_head(training_dataset, validation_dataset, embed_dim, max_epoch, 
                                                        batch_size, learning_rate, early_stop, device)
        else:
            raise NotImplementedError("wrong model head input")

        nfold_result_dict[n] = copy.deepcopy(result_dict)
        torch.save(best_model.state_dict(), saving_dir + head + "_fold{}_model.pt".format(n))

    save_data(saving_dir + head + "_result_dict.pkl", nfold_result_dict)


def nfold_evaluate_unirep_head_single_mutation(mutation_data_dir, representation_data_dir, saved_dir, mutation_file_name,
                                               pretrained_model, head, nfold, task, device):
    """
    evaluate n-fold saved models on the given task
    """

    test_metric_list = []
    representation_data_dict = load_data(representation_data_dir + pretrained_model + "_single_mutation_representaton_data_dict.pkl")
    data_nfold = np.array(load_nfold_data(mutation_data_dir, mutation_file_name, nfold)) # list of single folded split
    nfold_idx = np.tile(np.arange(nfold),2)

    if pretrained_model == "unirep-1900":
        embed_dim = 1900
    else:
        raise NotImplementedError("wrong pretrained model input")

    for n in range(nfold):

        testing_idx = np.array(nfold_idx[n+1])
        testing_data = data_nfold[testing_idx]
        testing_dataset = RepresentatonDataset(testing_data, representation_data_dict)

        if head == "linear":
            model = SimpleLinear(embed_dim, 1)
            model.load_state_dict(torch.load(saved_dir + head + "_fold{}_model.pt".format(n)))
        else:
            raise NotImplementedError("wrong head input")

        test_metric = evaluate_unirep_head_single_mutation(testing_dataset, model, task, device)
        test_metric_list.append(test_metric)

    if task == "binary":
        print("avg binary AUC: {}".format(np.mean(test_metric_list)))
        print("std binary AUC: {}".format(np.std(test_metric_list)))
    elif task == "score":
        print("avg spearman corr: {}".format(np.mean(test_metric_list)))
        print("std spearman corr: {}".format(np.std(test_metric_list)))
        
    return test_metric_list


def nfold_evaluate_head_multiple_mutations(mutation_data_dir, saved_dir,
                                           pretrained_model, head, nfold, score_computation, task, device):
    """
    evaluate n-fold saved models on the given task
    input representations of proteins are generated using Unirep model, not from saved representation dataset
    this is because it is not possible to save all possible double mutations 
    """

    test_metric_list = []
    testing_data = load_data(mutation_data_dir)
    testing_dataset = SequenceDataset(testing_data)

    print("loading pre-trained Unirep...")
    if pretrained_model == "unirep-1900":
        embed_dim = 1900
        unirep_model = UniRepModel(UniRepConfig())
        unirep_model.from_pretrained("babbler-1900")
        tokenizer = TAPETokenizer(vocab='unirep')
    else:
        raise NotImplementedError("wrong pretrained model input")
    print("loading pre-trained Unirep done")

    for n in range(nfold):

        if head == "linear":
            model = SimpleLinear(embed_dim, 1)
            model.load_state_dict(torch.load(saved_dir + head + "_fold{}_model.pt".format(n)))
        else:
            raise NotImplementedError("wrong head input")

        test_metric = evaluate_unirep_head_multiple_mutations(testing_dataset, unirep_model, model, tokenizer,
                                                              score_computation, task, device)
        test_metric_list.append(test_metric)

    if task == "binary":
        print("avg binary AUC: {}".format(np.mean(test_metric_list)))
        print("std binary AUC: {}".format(np.std(test_metric_list)))
    elif task == "score":
        print("avg spearman corr: {}".format(np.mean(test_metric_list)))
        print("std spearman corr: {}".format(np.std(test_metric_list)))
        
    return test_metric_list

