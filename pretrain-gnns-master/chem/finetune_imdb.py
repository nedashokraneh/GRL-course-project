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
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split
import pandas as pd
from TUDataset import TUDataset

import os
import shutil

from tensorboardX import SummaryWriter

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        
        ## manually add create node feature and edge attribute ([0, 0]) to IMDB data ## 
        num_nodes = batch.batch.shape[0]
        num_edges = batch.edge_index.shape[1]
        x = torch.zeros(size=[num_nodes, 2], dtype=torch.long).to(device)
        edge_attr = torch.zeros(size=[num_edges, 2], dtype=torch.long).to(device)

        pred = model(x, batch.edge_index, edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        # print(" ================= ")
        # print("TEST batch: \n{}\n".format(batch))
        # print("TEST batch.x: \n{}\n".format(batch.x))
        # print("TEST batch.edge_index: \n{}\n".format(batch.edge_index))
        # print("TEST batch.edge_attr: \n{}\n".format(batch.edge_attr))
        # print("TEST batch.batch: \n{}\n".format(batch.batch))
        # print("TEST pred: \n{}\n".format(pred))
        # print("TEST y: \n{}\n".format(y))

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        ## manually add create node feature and edge attribute ([0, 0]) to IMDB data ## 
        num_nodes = batch.batch.shape[0]
        num_edges = batch.edge_index.shape[1]
        x = torch.zeros(size=[num_nodes, 2], dtype=torch.long).to(device)
        edge_attr = torch.zeros(size=[num_edges, 2], dtype=torch.long).to(device)

        with torch.no_grad():
            pred = model(x, batch.edge_index, edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # print("TEST y_true[:,i]: {}".format(y_true[:, i]))
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0: # change '-1' to '0' because labels in IMDB-B are 0,1 instead of -1,1
            is_valid = y_true[:,i]**2 >= 0
            # print("TEST is_valid: {}".format(is_valid))
            # print("TEST y_scores[is_valid,i]: {}".format(y_scores[is_valid,i]))
            roc_list.append(roc_auc_score(y_true[is_valid,i], y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]



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
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'tox21', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 0, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    args = parser.parse_args()


    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if args.dataset == "imdb-b":
        num_tasks = 1
        dataset = TUDataset(root='data/imdb/binary', name='IMDB-BINARY')
    elif args.dataset == "imdb-m":
        num_tasks = 3 # 3 classes (??)
        dataset = TUDataset(root='data/imdb/multi', name='IMDB-MULTI')
    else:
        raise ValueError("Invalid dataset name.")    

    print(dataset)
    
    if args.split == "random":
        train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
    else:
        raise ValueError("Invalid split option.")

    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    model = GNN_graphpred(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling, gnn_type = args.gnn_type)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)
    
    model.to(device)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    if args.graph_pooling == "attention":
        model_param_group.append({"params": model.pool.parameters(), "lr":args.lr*args.lr_scale})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":args.lr*args.lr_scale})
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []


    if not args.filename == "":
        fname = 'runs/finetune_cls_runseed' + str(args.runseed) + '/' + args.filename
        #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train(args, model, device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, device, val_loader)
        test_acc = eval(args, model, device, test_loader)

        print("train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        if not args.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)

        print("")

    if not args.filename == "":
        writer.close()

if __name__ == "__main__":
    main()
