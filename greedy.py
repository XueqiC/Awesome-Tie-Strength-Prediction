from param_parser import parse_args
from utils import *
from model import  *
import os
import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
from learn import train, eval
import math
from sentence_transformers import SentenceTransformer
import time
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

start = time.time()
args = parse_args()
seed_everything(args.seed)

# args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.path = os.getcwd()
data = load_data(args)

# args.pr, args.de_sum, args.edge_tra, args.edge_tc, args.de_d, args.ecs, args.cn, args.uo = edge_scores(data)
# args.all_fea = np.column_stack((args.pr, args.de_sum, args.edge_tra, args.edge_tc, args.de_d, args.ecs, args.cn, args.uo))
res=[]
# num_classes = int(torch.unique(data.y).shape[0])
# prop_edge_index, prop_edge_attr = process_edge(data.edge_index, data.edge_attr)
# prop_edge_index, prop_edge_attr = prop_edge_index.to(args.device), prop_edge_attr.to(args.device)

if args.dataset == 'ba':    
    gre = np.loadtxt('./dataset/ba_greedy.csv', delimiter=',')
    gre = torch.tensor(gre, dtype=torch.long)
elif args.dataset == 'msg':
    gre = np.loadtxt('./dataset/msg_greedy.csv', delimiter=',')
    gre = torch.tensor(gre, dtype=torch.long)
elif args.dataset == 'tw':
    gre = np.loadtxt('./dataset/27_tw_greedy.csv', delimiter=',')
    gre = torch.tensor(gre, dtype=torch.long)
    
for i in range(args.runs):
    
    gre_np = gre.numpy()
    # test_np = np.load('dataset/tw/new_labels/label1.npy')[2,edge_idxs['test']]
    # test_np = np.load(f'dataset/tw/new/label{args.outfile}.npy')[2,:]#y_true
    test_np = data.y.numpy()
    weight = data.weight.numpy()
    
    acc = accuracy_score(test_np, gre_np)
    f1 = f1_score(test_np, gre_np)
    bacc = balanced_accuracy_score(test_np, gre_np)
    f1_macro = f1_score(test_np, gre_np, average='macro')
    
    y_true = test_np
    y_pred = gre_np
    w = [
        np.mean(weight[y_true == 0]) if np.any(y_true == 0) else 0,
        np.mean(weight[y_true == 1]) if np.any(y_true == 1) else 0,
        np.mean(weight[y_pred == 0]) if np.any(y_pred == 0) else 0,
        np.mean(weight[y_pred == 1]) if np.any(y_pred == 1) else 0
    ]
    
    result = {}   # Initialize an empty dictionary for this run's results
    result['test_acc'] = acc
    result['f1'] = f1
    result['bacc'] = bacc
    result['f1_macro'] = f1_macro
    result['w0'] = np.array(w)[0]
    result['w1'] = np.array(w)[1]
    result['w2'] = np.array(w)[2]
    result['w3'] = np.array(w)[3]
    res.append(result)

metrics = ['test_acc', 'f1', 'bacc', 'f1_macro', 'w0', 'w1', 'w2', 'w3']
# metrics = [ 'test_acc', 'f1']
means = {}
stds = {}
end = time.time()

for metric in metrics:
    values = [r[metric] for r in res]
    means[metric] = np.mean(values)
    stds[metric] = np.std(values)

# fi =[ 'cor_s', 'cor_acc']
# for metric in fi:   
#     values = [r[metric] for r in res]
#     means[metric] = np.mean(values, axis=0)
#     stds[metric] = np.std(values, axis=0)
#     print(means[metric])
    
metrics_output = []
for metric in metrics:
    metrics_output.append(r"{}: {:.3f} $\pm$ {:.3f}".format(metric, means[metric], stds[metric]))
    
output_string = f"Total time: {end-start:.2f} seconds," + f" #Strong: { data.y.sum()/len(data.y):.2f}, "+', '.join(metrics_output)

# Print the output to the console
print(output_string)