import pandas as pd
import seaborn as sns
import sys
sys.path.append("..")
sys.path.append("/home/nshokran/projects/GRL-course-project/LogME")
from LogME import LogME
import torch
import gcc
from gcc.datasets import (
    GRAPH_CLASSIFICATION_DSETS,
    GraphClassificationDataset,
    GraphClassificationDatasetLabeled,
    #LoadBalanceGraphDataset,
    NodeClassificationDataset,
    NodeClassificationDatasetLabeled,
    worker_init_fn,
)
from gcc.datasets.data_util import batcher
from gcc.models import GraphEncoder
from gcc.datasets import data_util

dataset_name = 'bbbp'

def print_model_args(args):
    for arg in vars(args):
        print(arg , " ", vars(args)[arg])

def test_moco(train_loader, model, opt):
    """
    one epoch training for moco
    """

    model.eval()

    emb_list = []
    for idx, batch in enumerate(train_loader):
        graph_q, graph_k = batch
        bsz = graph_q.batch_size
        graph_q.to(opt.device)
        graph_k.to(opt.device)

        with torch.no_grad():
            feat_q = model(graph_q)
            feat_k = model(graph_k)

        assert feat_q.shape == (bsz, opt.hidden_size)
        emb_list.append(((feat_q + feat_k) / 2).detach().cpu())
    return torch.cat(emb_list)


print("loading the pre-trained model")

checkpoint = torch.load('../saved/Pretrain_moco_False_dgl_gin_layer_5_lr_0.005_decay_1e-05_bsz_256_hid_64_samples_2000_nce_t_0.07_nce_k_32_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_0.999/current.pth', map_location="cpu")
#print_model_args(checkpoint["opt"])
args = checkpoint["opt"]
args.device = torch.device("cpu")
model = GraphEncoder(
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
        gnn_model=args.model,
        norm=args.norm,
        degree_input=True,
    )
model.load_state_dict(checkpoint["model"])

print("loading the dataset " + dataset_name)
train_dataset = GraphClassificationDataset(
            dataset=dataset_name,
            rw_hops=args.rw_hops,
            subgraph_size=args.subgraph_size,
            restart_prob=args.restart_prob,
            positional_embedding_size=args.positional_embedding_size,
        )
args.batch_size = len(train_dataset)
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=args.batch_size,
    collate_fn=batcher(),
    shuffle=False,
    num_workers=args.num_workers,
)

print("forwarding the data " + dataset_name + " to the pre-trained model.")
emb = test_moco(train_loader, model, args)

print("storing the results")
graphs_data = data_util.create_graph_classification_dataset(dataset_name)
graphs_labels = graphs_data.graph_labels
emb_df = pd.DataFrame(emb.numpy())
emb_df.columns = ['emb'+str(e+1) for e in range(emb_df.shape[1])]
emb_df['label'] = graphs_labels
emb_df.to_csv("results/"+dataset_name+".txt", sep = "\t", index = False)
