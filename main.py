from param_parser import parse_args
from utils import *
from model import  *
import os
import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
from learn import train, eval, evaltrain
import math
from sentence_transformers import SentenceTransformer
import time
import numpy as np
from sklearn.manifold import TSNE
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

# To ignore all user warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.*")


def run(data, loaders, model, classifier, optimizer, loss_fn, prop_edge_index, prop_edge_attr, args):
    best_val_f1= -math.inf
    best_epoch = 0
    patience = 0
    num_patience = 20
    
    for epoch in range(args.epochs):
        train(data, model, classifier, loaders['train'], optimizer, loss_fn, prop_edge_index, prop_edge_attr, args)

        if epoch % 5 == 0:
            val_acc, f1, bacc, _ , _,  = eval(data, model, classifier, loaders['val'], prop_edge_index, prop_edge_attr, args)

            if f1 > best_val_f1:
                # print('Epoch: {:03d}, Val Accuracy: {:.4f}, F1: {:.4f}'\
            #         .format(epoch, val_acc, f1))
                best_val_f1 = f1
                best_epoch = epoch
                patience = 0
                torch.save(model.state_dict(), f'./model/{args.dataset}/best_model.pt')
                torch.save(classifier.state_dict(), f'./model/{args.dataset}/best_classifier.pt')
                
            else:
                patience += 1
                if patience == num_patience:
                    break
    
    model.load_state_dict(torch.load(f'./model/{args.dataset}/best_model.pt'))
    classifier.load_state_dict(torch.load(f'./model/{args.dataset}/best_classifier.pt'))
    
    # _,_,_,_,_ = evaltrain(data, model, classifier, loaders['test'], prop_edge_index, prop_edge_attr,args)
    
    # test_acc, f1, bacc, f1_macro = eval(data, model, classifier, loaders['test'], prop_edge_index, args) #train_val_edge
    # test_acc, f1, bacc, f1_macro, cor_strong, cor_acc, w = eval(data, model, classifier, loaders['test'], prop_edge_index, prop_edge_attr,args) #train_val_edge
    
    test_acc, f1, bacc, f1_macro, w = eval(data, model, classifier, loaders['test'], prop_edge_index, prop_edge_attr, args) 
    
    end = time.time()
    # print('Time: {:.4f},  Best epoch: {}, Test Accuracy: {:.4f}, F1: {:.4f}, balanced accuracy: {:.4f}, Macro-F1: {:.4f} '\
    #     .format(end-start, int(best_epoch), test_acc, f1, bacc, f1_macro))
    # return test_acc, f1, bacc, f1_macro, cor_strong, cor_acc, w
    return test_acc, f1, bacc, f1_macro, w

def get_model(args, data):
    if args.model in ['Tran', 'GCN', 'GAT', 'SAGE', 'Cheb', 'MLP', 'MLP2']:
        if args.model == 'Tran':
            if data.edge_attr is None:
                model = TranConv(data.num_nodes, args.n_embed, args.n_hidden, edge_dim = None).to(args.device)
            else:
                model = TranConv(data.num_nodes, args.n_embed, args.n_hidden, data.edge_attr.shape[1]).to(args.device)
        elif args.model == 'GCN':
            model = GCNEncoder(data.num_nodes, args.n_embed, args.n_hidden).to(args.device)
        elif args.model == 'GAT':
            model = GATEncoder(data.num_nodes, args.n_embed, args.n_hidden).to(args.device)
        elif args.model == 'SAGE':
            model = SAGEEncoder(data.num_nodes, args.n_embed, args.n_hidden).to(args.device)
        elif args.model == 'Cheb':
            model = ChebEncoder(data.num_nodes, args.n_embed, args.n_hidden).to(args.device)
        elif args.model == 'MLP':
            model = mlp(data.num_nodes, args.n_embed, args.n_hidden).to(args.device)
        elif args.model == 'MLP2':
            model = mlp(data.num_nodes, args.n_embed, args.n_hidden).to(args.device)
        
        if args.model in ['Tran', 'MLP2']:
            classifier = Classifier2(torch.unique(data.y).shape[0], args).to(args.device)
        else:
            if data.edge_attr is None:
                classifier = Classifier2(torch.unique(data.y).shape[0], args).to(args.device)
            else:
                classifier = Classifier(data.edge_attr.shape[1], torch.unique(data.y).shape[0], args).to(args.device)
    else:
        raise ValueError("Invalid model type")
    
    return model, classifier


if __name__ == '__main__':
    start = time.time()
    args = parse_args()
    seed_everything(args.seed)
            
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.path = os.getcwd()
    data = load_data(args)

    res=[]
    num_classes = int(torch.unique(data.y).shape[0])
    prop_edge_index, prop_edge_attr = process_edge(data.edge_index, data.edge_attr)
    prop_edge_index, prop_edge_attr = prop_edge_index.to(args.device), prop_edge_attr.to(args.device)
    
    # args.edge_tra, args.edge_tc, args.cn, args.uo = edge_scores(data)
    # args.all_fea = np.column_stack((args.edge_tra, args.edge_tc, args.cn, args.uo))
    # data.all_fea = args.all_fea
    
    if args.model == 'rf':
        # data.uo = torch.tensor(un_overlap(data, args))
        # data.ecs = ec(data)
        args.pr, args.de_sum, args.edge_tra, args.edge_tc, args.de_d, args.ecs, args.cn, args.uo = edge_scores(data)
        args.all_fea = np.column_stack((args.pr, args.de_sum, args.edge_tra, args.edge_tc, args.de_d, args.ecs, args.cn, args.uo))
        # args.edge_tra, args.edge_tc, args.cn = edge_scores(data)
        # args.all_fea = np.column_stack((args.edge_tra, args.edge_tc, args.cn))
        data.all_fea = args.all_fea
        
    for i in range(args.runs):
        if args.task == 1: 
            edge_idxs = random_split_edge(data, args)
        elif args.task == 2:
            edge_idxs = random_split_node(data, args)
            
        if args.rw == 1:
            num_classes = int(torch.unique(data.y).shape[0])
            train_y = data.y[edge_idxs['train']]
            args.reweight = cal_reweight(train_y)
            args.reweight = args.reweight.to(args.device)
        
        if args.model == 'rf':
            test_acc, f1, bacc, f1_macro, cor_s, cor_acc, w = rf(data, edge_idxs)
        else:
            data = data.to(args.device) 
            split_data = {key: EdgeDataset(edge_idxs[key]) for key in edge_idxs}
            loaders = {}
            for key in split_data:
                if key == 'train':
                    shuffle = True
                else:
                    shuffle = False
                loaders[key] = DataLoader(split_data[key], batch_size = args.batch_size, \
                                        shuffle = shuffle, collate_fn = split_data[key].collate_fn)
            
            model, classifier = get_model(args, data)
            loss_fn = nn.CrossEntropyLoss(reduction='none')
            optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=args.lr)
            
            # test_acc, f1, bacc, f1_macro, cor_s, cor_acc, w =run(data, loaders, model, classifier, optimizer, loss_fn, prop_edge_index, prop_edge_attr, args)
            test_acc, f1, bacc, f1_macro, w =run(data, loaders, model, classifier, optimizer, loss_fn, prop_edge_index, prop_edge_attr, args)
            
        result = {}   # Initialize an empty dictionary for this run's results
        result['test_acc'] = test_acc
        result['f1'] = f1
        result['bacc'] = bacc
        result['f1_macro'] = f1_macro
        result['w0'] = np.array(w)[0]
        result['w1'] = np.array(w)[1]
        result['w2'] = np.array(w)[2]
        result['w3'] = np.array(w)[3]

        # result['cor_s'] = np.array(cor_s)
        # result['cor_acc'] = np.array(cor_acc)
        res.append(result)
        data = data.to('cpu')
    
    # metrics = ['test_acc', 'f1', 'bacc', 'f1_macro', 'w0', 'w1', 'w2', 'w3']
    metrics = ['test_acc', 'f1_macro', 'w0', 'w1', 'w2', 'w3']
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
