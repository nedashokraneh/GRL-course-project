import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import sys
sys.path.append("..")
import torch
import gcc
from gcc.datasets import (
    GRAPH_CLASSIFICATION_DSETS,
    GraphClassificationDataset,
    GraphClassificationDatasetLabeled,
    LoadBalanceGraphDataset,
    NodeClassificationDataset,
    NodeClassificationDatasetLabeled,
    worker_init_fn,
)
from gcc.datasets.data_util import batcher
from gcc.models import GraphEncoder
from gcc.datasets import data_util
from collections import defaultdict, namedtuple

from gcc.utils.splitter import random_split
from gcc.datasets.data_util import batcher, labeled_batcher
from gcc.contrastive.memory_moco import MemoryMoCo
from torch import nn
from gcc.utils.misc import AverageMeter, adjust_learning_rate, warmup_linear

from sklearn.metrics import accuracy_score, f1_score

epoch_num = 50
run_num = 10
dataset_name = 'imdb-multi'

def train_finetune(
    epoch,
    train_loader,
    model,
    output_layer,
    criterion,
    optimizer,
    output_layer_optimizer,
    opt,
):
    """
    one epoch training for moco
    """
    n_batch = len(train_loader)
    model.train()
    output_layer.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    f1_meter = AverageMeter()
    epoch_loss_meter = AverageMeter()
    epoch_f1_meter = AverageMeter()
    prob_meter = AverageMeter()
    graph_size = AverageMeter()
    max_num_nodes = 0
    max_num_edges = 0

    end = time.time()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        graph_q, y = batch

        graph_q
        y = y

        bsz = graph_q.batch_size

        # ===================forward=====================

        feat_q = model(graph_q)

        assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)
        out = output_layer(feat_q)

        loss = criterion(out, y)

        # ===================backward=====================
        optimizer.zero_grad()
        output_layer_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1)
        torch.nn.utils.clip_grad_value_(output_layer.parameters(), 1)
        global_step = epoch * n_batch + idx
        lr_this_step = opt.learning_rate * warmup_linear(
            global_step / (opt.epochs * n_batch), 0.1
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_this_step
        for param_group in output_layer_optimizer.param_groups:
            param_group["lr"] = lr_this_step
        optimizer.step()
        output_layer_optimizer.step()

        preds = out.argmax(dim=1)
        f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average="micro")

        # ===================meters=====================
        f1_meter.update(f1, bsz)
        epoch_f1_meter.update(f1, bsz)
        loss_meter.update(loss.item(), bsz)
        epoch_loss_meter.update(loss.item(), bsz)
        graph_size.update(graph_q.number_of_nodes() / bsz, bsz)
        max_num_nodes = max(max_num_nodes, graph_q.number_of_nodes())
        max_num_edges = max(max_num_edges, graph_q.number_of_edges())

        #torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            mem = psutil.virtual_memory()
            print(
                "Train: [{0}][{1}/{2}]\t"
                "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "loss {loss.val:.3f} ({loss.avg:.3f})\t"
                "f1 {f1.val:.3f} ({f1.avg:.3f})\t"
                "GS {graph_size.val:.3f} ({graph_size.avg:.3f})\t"
                "mem {mem:.3f}".format(
                    epoch,
                    idx + 1,
                    n_batch,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=loss_meter,
                    f1=f1_meter,
                    graph_size=graph_size,
                    mem=mem.used / 1024 ** 3,
                )
            )
    return epoch_loss_meter.avg, epoch_f1_meter.avg

def test_finetune(epoch, valid_loader, model, output_layer, criterion, opt):
    n_batch = len(valid_loader)
    model.eval()
    output_layer.eval()

    epoch_loss_meter = AverageMeter()
    epoch_f1_meter = AverageMeter()

    for idx, batch in enumerate(valid_loader):
        graph_q, y = batch

        bsz = graph_q.batch_size

        # ===================forward=====================

        with torch.no_grad():
            feat_q = model(graph_q)
            assert feat_q.shape == (graph_q.batch_size, opt.hidden_size)
            out = output_layer(feat_q)
        loss = criterion(out, y)

        preds = out.argmax(dim=1)
        f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average="micro")

        # ===================meters=====================
        epoch_loss_meter.update(loss.item(), bsz)
        epoch_f1_meter.update(f1, bsz)

    global_step = (epoch + 1) * n_batch

    return epoch_loss_meter.avg, epoch_f1_meter.avg



checkpoint = torch.load('saved/Pretrain_moco_False_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_256_hid_64_samples_2000_nce_t_0.07_nce_k_32_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.999/current.pth', map_location="cpu")
args = checkpoint["opt"]
args.device = torch.device("cpu")


dataset = GraphClassificationDatasetLabeled(
                dataset=dataset_name,
                rw_hops=args.rw_hops,
                subgraph_size=args.subgraph_size,
                restart_prob=args.restart_prob,
                positional_embedding_size=args.positional_embedding_size,
            )
labels = dataset.dataset.graph_labels

for r in range(1,run_num+1):

    train_dataset, valid_dataset, test_dataset = random_split(dataset, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
    train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            collate_fn=labeled_batcher(),
            shuffle=True,
            num_workers=args.num_workers,
            worker_init_fn=None,
        )
    valid_loader = torch.utils.data.DataLoader(
                dataset=valid_dataset,
                batch_size=args.batch_size,
                collate_fn=labeled_batcher(),
                num_workers=args.num_workers,
        )
    test_loader = valid_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size,
                collate_fn=labeled_batcher(),
                num_workers=args.num_workers,
        )


    ########################supervised from scratch#################################
    model, model_ema = [
            GraphEncoder(
                positional_embedding_size=args.positional_embedding_size,
                max_node_freq=args.max_node_freq,
                max_edge_freq=args.max_edge_freq,
                max_degree=args.max_degree,
                freq_embedding_size=args.freq_embedding_size,
                degree_embedding_size=args.degree_embedding_size,
                output_dim=args.hidden_size,
                node_hidden_dim=args.hidden_size,
                edge_hidden_dim=args.hidden_size,
                num_layers=args.num_layer,
                num_step_set2set=args.set2set_iter,
                num_layer_set2set=args.set2set_lstm_layer,
                norm=args.norm,
                gnn_model=args.model,
                degree_input=True,
            )
            for _ in range(2)
        ]
    output_layer = nn.Linear(
                in_features=args.hidden_size, out_features=dataset.dataset.num_labels
            )
    contrast = MemoryMoCo(
            args.hidden_size, None, args.nce_k, args.nce_t, use_softmax=True
        )
    criterion = nn.CrossEntropyLoss()
    output_layer_optimizer = torch.optim.Adam(
                output_layer.parameters(),
                lr=args.learning_rate,
                betas=(args.beta1, args.beta2),
                weight_decay=args.weight_decay,
            )

    def clear_bn(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.reset_running_stats()

    model.apply(clear_bn)
    optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                betas=(args.beta1, args.beta2),
                weight_decay=args.weight_decay,
            )

    train_losses = []
    train_f1s = []
    valid_losses = []
    valid_f1s = []
    test_losses = []
    test_f1s = []
    print('training from scratch, run {}'.format(r))
    for epoch in range(1, epoch_num+1):

            adjust_learning_rate(epoch, args, optimizer)


            time1 = time.time()
            train_loss, train_f1 = train_finetune(
                epoch,
                train_loader,
                model,
                output_layer,
                criterion,
                optimizer,
                output_layer_optimizer,
                args,
            )
            valid_loss, valid_f1 = test_finetune(
                epoch,
                valid_loader,
                model,
                output_layer,
                criterion,
                args
            )
            test_loss, test_f1 = test_finetune(
                epoch,
                test_loader,
                model,
                output_layer,
                criterion,
                args
            )

            time2 = time.time()
            train_losses.append(train_loss)
            train_f1s.append(train_f1)
            valid_losses.append(valid_loss)
            valid_f1s.append(valid_f1)
            test_losses.append(test_loss)
            test_f1s.append(test_f1)
    ###############################fine-tuning from pre-trained model###############
    model, model_ema = [
            GraphEncoder(
                positional_embedding_size=args.positional_embedding_size,
                max_node_freq=args.max_node_freq,
                max_edge_freq=args.max_edge_freq,
                max_degree=args.max_degree,
                freq_embedding_size=args.freq_embedding_size,
                degree_embedding_size=args.degree_embedding_size,
                output_dim=args.hidden_size,
                node_hidden_dim=args.hidden_size,
                edge_hidden_dim=args.hidden_size,
                num_layers=args.num_layer,
                num_step_set2set=args.set2set_iter,
                num_layer_set2set=args.set2set_lstm_layer,
                norm=args.norm,
                gnn_model=args.model,
                degree_input=True,
            )
            for _ in range(2)
        ]
    output_layer = nn.Linear(
                in_features=args.hidden_size, out_features=dataset.dataset.num_labels
            )
    contrast = MemoryMoCo(
            args.hidden_size, None, args.nce_k, args.nce_t, use_softmax=True
        )
    criterion = nn.CrossEntropyLoss()
    output_layer_optimizer = torch.optim.Adam(
                output_layer.parameters(),
                lr=args.learning_rate,
                betas=(args.beta1, args.beta2),
                weight_decay=args.weight_decay,
            )

    def clear_bn(m):
        classname = m.__class__.__name__
        if classname.find("BatchNorm") != -1:
            m.reset_running_stats()
    model.apply(clear_bn)
    optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                betas=(args.beta1, args.beta2),
                weight_decay=args.weight_decay,
            )
    model.load_state_dict(checkpoint["model"])
    contrast.load_state_dict(checkpoint["contrast"])

    ft_train_losses = []
    ft_train_f1s = []
    ft_valid_losses = []
    ft_valid_f1s = []
    ft_test_losses = []
    ft_test_f1s = []
    print('training from pre-trained, run {}'.format(r))
    for epoch in range(1,epoch_num+1):

            adjust_learning_rate(epoch, args, optimizer)

            time1 = time.time()
            train_loss, train_f1 = train_finetune(
                epoch,
                train_loader,
                model,
                output_layer,
                criterion,
                optimizer,
                output_layer_optimizer,
                args,
            )
            valid_loss, valid_f1 = test_finetune(
                epoch,
                valid_loader,
                model,
                output_layer,
                criterion,
                args
            )
            test_loss, test_f1 = test_finetune(
                epoch,
                test_loader,
                model,
                output_layer,
                criterion,
                args
            )

            time2 = time.time()
            ft_train_losses.append(train_loss)
            ft_train_f1s.append(train_f1)
            ft_valid_losses.append(valid_loss)
            ft_valid_f1s.append(valid_f1)
            ft_test_losses.append(test_loss)
            ft_test_f1s.append(test_f1)

    losses = pd.DataFrame(columns = ['split', 'training_type', 'epoch', 'loss'])
    losses = pd.concat([losses, pd.DataFrame({'split': 'train', 'training_type': 'No pre-training', 'epoch': np.arange(1,epoch_num+1), 'loss': train_losses})])
    losses = pd.concat([losses, pd.DataFrame({'split': 'valid', 'training_type': 'No pre-training', 'epoch': np.arange(1,epoch_num+1), 'loss': valid_losses})])
    losses = pd.concat([losses, pd.DataFrame({'split': 'train', 'training_type': 'Fine-tuning', 'epoch': np.arange(1,epoch_num+1), 'loss': ft_train_losses})])
    losses = pd.concat([losses, pd.DataFrame({'split': 'valid', 'training_type': 'Fine-tuning', 'epoch': np.arange(1,epoch_num+1), 'loss': ft_valid_losses})])
    f1s = pd.DataFrame(columns = ['split', 'training_type', 'epoch', 'f1'])
    f1s = pd.concat([f1s, pd.DataFrame({'split': 'train', 'training_type': 'No pre-training', 'epoch': np.arange(1,epoch_num+1), 'f1': train_f1s})])
    f1s = pd.concat([f1s, pd.DataFrame({'split': 'valid', 'training_type': 'No pre-training', 'epoch': np.arange(1,epoch_num+1), 'f1': valid_f1s})])
    f1s = pd.concat([f1s, pd.DataFrame({'split': 'train', 'training_type': 'Fine-tuning', 'epoch': np.arange(1,epoch_num+1), 'f1': ft_train_f1s})])
    f1s = pd.concat([f1s, pd.DataFrame({'split': 'valid', 'training_type': 'Fine-tuning', 'epoch': np.arange(1,epoch_num+1), 'f1': ft_valid_f1s})])


    params = {'legend.fontsize': 12,
             'axes.labelsize': 16,
             'axes.titlesize':20,
             'xtick.labelsize': 12,
             'ytick.labelsize': 12}
    plt.rcParams.update(params)
    fig, axs = plt.subplots(1,2, figsize = (10,4))
    sns.lineplot(data = losses, x = 'epoch', y = 'loss', hue = 'split', style = 'training_type', ax = axs[0])
    axs[0].set(xlabel='epoch',ylabel='loss')
    axs[0].legend_.remove()
    sns.lineplot(data = f1s, x = 'epoch', y = 'f1', hue = 'split', style = 'training_type', ax = axs[1])
    axs[1].set(xlabel='epoch',ylabel='f1 score')
    axs[1].legend(bbox_to_anchor=(1.01, 1), ncol=1)
    fig.tight_layout(pad=1.0)
    fig_name = '{}_GCC_run{}'.format(dataset_name,r)
    fig.savefig('/media/nedooshki/f4f0aea6-900a-437f-82e1-238569330477/GRL-course-project/results/plots/{}.png'.format(fig_name), dpi=300)



    outF = open("{}_finetuning_result.txt".format(dataset_name), "a")
    outF.write('fine_tuned f1 score for run {} is {}\u00B1{}\n'.format(r,round(np.mean(ft_test_f1s[-10:]),2),round(np.sqrt(np.var(ft_test_f1s[-10:])),2)))
    outF.write('supervised f1 score for run {} is {}\u00B1{}\n'.format(r,round(np.mean(test_f1s[-10:]),2),round(np.sqrt(np.var(test_f1s[-10:])),2)))
    outF.close()
