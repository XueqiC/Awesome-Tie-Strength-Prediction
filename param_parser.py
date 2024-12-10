import argparse

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tw', help='select from [ba: bitcoin_alpha, msg, tw: twitter]')
    parser.add_argument('--task', type=int, default=2, help='0: unsupervised learning, 1: weakly supervised learning (random links), 2: weakly supervised learning (hub node) ')
    parser.add_argument('--seed', type=int, default=1028)
    
    parser.add_argument('--setting', type=int, default=1, help='different tie strength definition')
    parser.add_argument('--thre', type=float, default=4, help='threshold')
    parser.add_argument('--tw', type=float, default=2e7, help='time window for msg')
    
    parser.add_argument('--train_ratio', type=float, default=0.05)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--label_ratio', type=float, default=1.0)
    parser.add_argument('--runs', type=int, default=3)
    
    parser.add_argument('--batch_size', type=int, default=8192)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model', type=str, default='rf', help='select from [GCN, GAT, Cheb, MLP, MLP2, rf, Tran]')
    
    parser.add_argument('--n_embed', type=int, default=64)
    parser.add_argument('--n_hidden', type=int, default=64)
    parser.add_argument('--n_out', type=int, default=32)
    parser.add_argument('--heads', type=int, default=2)
    
    parser.add_argument('--training', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--method', type=str, default='concat', help='select from [concat, hadamard, add, or average]')
    
    parser.add_argument('--rw', type=int, default=1, help='0: no reweighting, 1: reweighting')
    
    parser.add_argument('--outfile', type=int, default=1)

    return parser.parse_args()