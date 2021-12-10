# Code based on pretrain_supervised.py
#
# To train downstream tasks from scratch (BBBP, BACE)

import argparse

from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import numpy as np

from model import GNN, GNN_graphpred
from sklearn.metrics import roc_auc_score, f1_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

from TUDataset import TUDataset

from tensorboardX import SummaryWriter

# criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(args, model, device, loader, optimizer, criterion):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        
        if args.dataset in ['imdb-b', 'imdb-m']:
            ## manually add create node feature and edge attribute ([0, 0]) to IMDB data ## 
            num_nodes = batch.batch.shape[0]
            num_edges = batch.edge_index.shape[1]
            x = torch.zeros(size=[num_nodes, 2], dtype=torch.long).to(device)
            edge_attr = torch.zeros(size=[num_edges, 2], dtype=torch.long).to(device)

            pred = model(x, batch.edge_index, edge_attr, batch.batch)
        else:
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        if args.evaluation == 'auc':
            y = batch.y.view(pred.shape).to(torch.float64)
        elif args.evaluation == 'f1':
            y = batch.y.to(torch.float64)

        #Whether y is non-null or not.
        if args.dataset in ['bbbp', 'bace']:
            is_valid = y**2 > 0
        else:
            is_valid = y**2 >= 0
        #Loss matrix
        if args.evaluation == 'auc':
            loss_mat = criterion(pred.double(), (y+1)/2)
            #loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
                
            optimizer.zero_grad()
            loss = torch.sum(loss_mat)/torch.sum(is_valid)
            loss.backward()

            optimizer.step()

        elif args.evaluation == 'f1':
            if args.dataset in ['bbbp', 'bace']:
                loss = criterion(pred.double(), ((y+1)/2).long())
            elif args.dataset in ['imdb-b', 'imdb-m']:
                loss = criterion(pred.double(), y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def eval(args, model, device, loader, normalized_weight=None):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            if args.dataset in ['imdb-b', 'imdb-m']:
                ## manually add create node feature and edge attribute ([0, 0]) to IMDB data ## 
                num_nodes = batch.batch.shape[0]
                num_edges = batch.edge_index.shape[1]
                x = torch.zeros(size=[num_nodes, 2], dtype=torch.long).to(device)
                edge_attr = torch.zeros(size=[num_edges, 2], dtype=torch.long).to(device)

                pred = model(x, batch.edge_index, edge_attr, batch.batch)
            else:
                pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        if args.evaluation == 'auc':
            y_true.append(batch.y.view(pred.shape).cpu())
            y_scores.append(pred.cpu())
        else:
            y_true.append(batch.y)
            y_scores.append(pred)

    if args.evaluation == 'auc':
        y_true = torch.cat(y_true, dim = 0).numpy()
        y_scores = torch.cat(y_scores, dim = 0).numpy()

        roc_list = []
        weight = []
        for i in range(y_true.shape[1]):
            #AUC is only defined when there is at least one positive data.
            if args.dataset in ['bbbp', 'bace']:
                if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
                    is_valid = y_true[:,i]**2 > 0
                    roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
                    # weight.append(normalized_weight[i])
            else:
                if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                    is_valid = y_true[:,i]**2 >= 0
                    roc_list.append(roc_auc_score(y_true[is_valid,i], y_scores[is_valid,i]))
                    # weight.append(normalized_weight[i])

        if len(roc_list) < y_true.shape[1]:
            print("Some target is missing!")
            print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

        # weight = np.array(weight)
        roc_list = np.array(roc_list)

        # return weight.dot(roc_list)
        return sum(roc_list)/len(roc_list) 
    elif args.evaluation == 'f1':

        y_true = torch.cat(y_true, dim=0)
        y_scores = torch.cat(y_scores, dim=0)
        preds = y_scores.argmax(dim=1)
        # print("TEST y_true (-1, 1 ==> 0, 1): {}".format((y_true.cpu().numpy()+1)/2))
        # print("TEST preds: {}".format(preds))
        if args.dataset in ['bbbp', 'bace']:
            f1 = f1_score((y_true.cpu().numpy()+1)/2, preds.cpu().numpy(), average='micro')
        elif args.dataset in ['imdb-b', 'imdb-m']:
            f1 = f1_score(y_true.cpu().numpy(), preds.cpu().numpy(), average='micro')

        return f1
    else:
        raise ValueError("Evaluation metric not defined.")


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'chembl_filtered', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--num_workers', type=int, default = 8, help='number of workers for dataset loading')
    parser.add_argument('--evaluation', type=str, default='auc', help='evaluation metric: auc, f1')
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    if args.evaluation == 'auc':
        criterion = nn.BCEWithLogitsLoss(reduction = "none")
    elif args.evaluation == 'f1':
        criterion = nn.CrossEntropyLoss() 

    #Bunch of classification tasks
    if args.dataset == "bbbp":
        if args.evaluation == 'auc':
            num_tasks = 1
        elif args.evaluation == 'f1':
            num_tasks = 2
    elif args.dataset == 'bace':
        if args.evaluation == 'auc':
            num_tasks = 1
        elif args.evaluation == 'f1':
            num_tasks = 2
    elif args.dataset == 'imdb-b':
        if args.evaluation == 'auc':
            num_tasks = 1
        elif args.evaluation == 'f1':
            num_tasks = 2
    elif args.dataset == 'imdb-m':
        num_tasks = 3
    else:
        raise ValueError("Invalid dataset name.")

    #set up dataset
    if args.dataset in ['bbbp', 'bace']:

        dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

        # loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
        smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    else:
        if args.dataset == 'imdb-b':
            dataset = TUDataset(root='data/imdb/binary', name='IMDB-BINARY')
        elif args.dataset == 'imdb-m':
            dataset = TUDataset(root='data/imdb/multi', name='IMDB-MULTI')
        
        # random split # 
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = 0)

    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file + ".pth")
    
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)  
    print(optimizer)

    # val_acc_list = []
    # test_acc_list = []
    # train_acc_list = []

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
    
        train(args, model, device, train_loader, optimizer, criterion)

        print("====Evaluation")
        train_acc = eval(args, model, device, train_loader)
        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)

        print("train: %f val: %f test: %f"%(train_acc, val_acc, test_acc))

        # val_acc_list.append(val_acc)
        # test_acc_list.append(test_acc)
        # train_acc_list.append(train_acc)

        # if not args.filename == "":
        #     writer.add_scalar('data/train auc', train_acc, epoch)
        #     writer.add_scalar('data/val auc', val_acc, epoch)
        #     writer.add_scalar('data/test auc', test_acc, epoch)

    if not args.output_model_file == "":
        torch.save(model.gnn.state_dict(), args.output_model_file + ".pth")

    if not args.filename == "":
        writer.close()

if __name__ == "__main__":
    main()
