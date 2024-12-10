from utils import *
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix
import time
import math
import networkx as nx


def train(data, model, classifier, loader, optimizer, loss_fn, prop_edge_index, prop_edge_attr, args):

    model.train()
    classifier.train()
    losses = []

    for batch in loader:
        optimizer.zero_grad()
        idx1, idx2 = batch[0], batch[1] #global and local index
        idx1, idx2 = idx1.to(args.device), idx2.to(args.device)
        
        if args.model == 'Tran':
            x = model(prop_edge_index, prop_edge_attr, args)
            edge_emb1 = edge_emb(x, data.edge_index, idx1, method=args.method)
            out = classifier(edge_emb1)
        elif args.model == 'MLP2':
            x = model(prop_edge_index, args)
            edge_emb1 = edge_emb(x, data.edge_index, idx1, method=args.method)
            out = classifier(edge_emb1)
        else:
            x = model(prop_edge_index, args)
            edge_emb1 = edge_emb(x, data.edge_index, idx1, method=args.method)
            if data.edge_attr is None:
                out = classifier(edge_emb1)
            else:
                attr_emb = data.edge_attr[idx1]
                out = classifier(edge_emb1, attr_emb)
                
        if args.rw == 1:
            loss0 = loss_fn(out, data.y[idx1].view(-1))
            loss = (loss0 * args.reweight[data.y[idx1].view(-1)]).mean()
        else:
            loss = loss_fn(out, data.y[idx1].view(-1)).mean()
        
        loss.backward()
        losses.append(loss.mean().item())
            
        optimizer.step()

    return np.mean(losses)


@torch.no_grad()
def eval(data, model, classifier, loader, prop_edge_index, prop_edge_attr,args):
    model.eval()
    classifier.eval()
    # all_feature = torch.tensor(args.all_fea).to(args.device)

    y_true, y_pred, all_fea, strong, weight = [], [], [], [], []
    index = []
    for batch in loader:
        idx1, idx2 = batch[0], batch[1]
        idx1, idx2 = idx1.to(args.device), idx2.to(args.device)
        
        if args.model == 'Tran':
            x = model(prop_edge_index, prop_edge_attr, args)
            edge_emb1 = edge_emb(x, data.edge_index, idx1, method=args.method)
            out = classifier(edge_emb1)
        elif args.model == 'MLP2':
            x = model(prop_edge_index, args)
            edge_emb1 = edge_emb(x, data.edge_index, idx1, method=args.method)
            out = classifier(edge_emb1)
        else:
            x = model(prop_edge_index, args)
            edge_emb1 = edge_emb(x, data.edge_index, idx1, method=args.method)
            if data.edge_attr is None:
                out = classifier(edge_emb1)
            else:
                attr_emb = data.edge_attr[idx1]
                out = classifier(edge_emb1, attr_emb)
                
        index.extend(idx1.cpu().numpy().tolist())
        y_true.extend(data.y[idx1].view(-1).cpu().numpy().tolist())
        y_pred.extend(out.argmax(dim=1).cpu().numpy().tolist())
        weight.extend(data.weight[idx1].cpu().numpy().tolist())
        # all_fea.extend(all_feature[idx1,:].cpu().numpy())
        strong.extend(out[:,1].cpu().numpy().tolist())
    
    #save index and y_pred
    # np.save('index.npy', index)
    # np.save('y_pred.npy', y_pred)
    
    # all_fea=np.array(all_fea)
    # cor_strong = np.zeros(all_fea.shape[1])
    # for i in range(all_fea.shape[1]):
    #     feature = all_fea[:, i]
    #     if np.std(feature) == 0 or np.std(strong) == 0:
    #         correlation = 0  # Correlation is undefined when there's no variance
    #     else:
    #         correlation = np.corrcoef(feature, strong)[0, 1]
    #     cor_strong[i] = correlation
        
    # cor_acc = np.zeros(all_fea.shape[1])
    # for i in range(all_fea.shape[1]):
    #     feature = all_fea[:, i]
    #     if np.std(feature) == 0 or np.std(y_true) == 0:
    #         correlation = 0  # Correlation is undefined when there's no variance
    #     else:
    #         correlation = np.corrcoef(feature, y_true)[0, 1]
    #     cor_acc[i] = correlation

    accuracy = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    w_f1 = f1_score(y_true, y_pred, average = 'weighted')
    f1_macro = f1_score(y_true, y_pred, average = 'macro')
    # f1_micro = f1_score(y_true, y_pred, average = 'micro')
    # confusion = confusion_matrix(y_true, y_pred)
    
    # edge_index = data.edge_index[index]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    weight = np.array(weight)
    
    # G = nx.Graph()
    # edge_index_cpu = edge_index.cpu()  # Move tensor to CPU
    # G.add_edges_from(edge_index_cpu.permute(1, 0).numpy().tolist())
    # pos = nx.spring_layout(G)
    
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # # Prepare figure
    # def plot_edges(ax, labels, title):
    #     for i, edge in enumerate(edge_index.permute(1, 0).numpy()):
    #         start, end = pos[edge[0]], pos[edge[1]]
    #         color = 'red' if labels[i] == 0 else 'blue'
    #         ax.plot([start[0], end[0]], [start[1], end[1]], color=color, alpha=0.7)
    #     ax.set_title(title)
    #     ax.axis('off')

    # # Plot true labels
    # plot_edges(axes[0], y_true, 'True Labels')

    # # Plot predicted labels
    # plot_edges(axes[1], y_pred, 'Predicted Labels')

    # # Save the plot
    # plt.savefig('edge_label_comparison.png')
    # plt.show()

    # Calculate average weights for each label in y_true and y_pred
    avg_w = [
        np.mean(weight[y_true == 0]) if np.any(y_true == 0) else 0,
        np.mean(weight[y_true == 1]) if np.any(y_true == 1) else 0,
        np.mean(weight[y_pred == 0]) if np.any(y_pred == 0) else 0,
        np.mean(weight[y_pred == 1]) if np.any(y_pred == 1) else 0
    ]
    
    # return accuracy, bacc, w_f1, f1_macro, f1_micro, confusion, y_true, y_pred
    # return accuracy, w_f1, bacc, f1_macro, cor_strong, cor_acc, avg_w
    return accuracy, w_f1, bacc, f1_macro, avg_w

@torch.no_grad()
def evaltrain(data, model, classifier, loader, prop_edge_index, prop_edge_attr,args):
    model.eval()
    classifier.eval()
    all_feature = torch.tensor(args.all_fea).to(args.device)

    y_true, y_pred, all_fea, strong, weight = [], [], [], [], []
    index = []
    for batch in loader:
        idx1, idx2 = batch[0], batch[1]
        idx1, idx2 = idx1.to(args.device), idx2.to(args.device)
        
        if args.model == 'Tran':
            x = model(prop_edge_index, prop_edge_attr, args)
            edge_emb1 = edge_emb(x, data.edge_index, idx1, method=args.method)
            out = classifier(edge_emb1)
        elif args.model == 'MLP2':
            x = model(prop_edge_index, args)
            edge_emb1 = edge_emb(x, data.edge_index, idx1, method=args.method)
            out = classifier(edge_emb1)
        else:
            x = model(prop_edge_index, args)
            edge_emb1 = edge_emb(x, data.edge_index, idx1, method=args.method)
            if data.edge_attr is None:
                out = classifier(edge_emb1)
            else:
                attr_emb = data.edge_attr[idx1]
                out = classifier(edge_emb1, attr_emb)

        index.extend(idx1.cpu().numpy().tolist())
        y_true.extend(data.y[idx1].view(-1).cpu().numpy().tolist())
        y_pred.extend(out.argmax(dim=1).cpu().numpy().tolist())
        weight.extend(data.weight[idx1].cpu().numpy().tolist())
        all_fea.extend(all_feature[idx1,:].cpu().numpy())
        strong.extend(out[:,1].cpu().numpy().tolist())
        
    all_fea=np.array(all_fea)
    # cor_strong = np.zeros(all_fea.shape[1])
    # for i in range(all_fea.shape[1]):
    #     feature = all_fea[:, i]
    #     if np.std(feature) == 0 or np.std(strong) == 0:
    #         correlation = 0  # Correlation is undefined when there's no variance
    #     else:
    #         correlation = np.corrcoef(feature, strong)[0, 1]
    #     cor_strong[i] = correlation
        
    # cor_acc = np.zeros(all_fea.shape[1])
    # for i in range(all_fea.shape[1]):
    #     feature = all_fea[:, i]
    #     if np.std(feature) == 0 or np.std(y_true) == 0:
    #         correlation = 0  # Correlation is undefined when there's no variance
    #     else:
    #         correlation = np.corrcoef(feature, y_true)[0, 1]
    #     cor_acc[i] = correlation

    accuracy = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    w_f1 = f1_score(y_true, y_pred, average = 'weighted')
    f1_macro = f1_score(y_true, y_pred, average = 'macro')
    # f1_micro = f1_score(y_true, y_pred, average = 'micro')
    # confusion = confusion_matrix(y_true, y_pred)
    
    edge_index = data.edge_index[index]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    weight = np.array(weight)
    
    plot_acc_fea(y_true, y_pred, all_fea, args, n_bins=6)
    # plot_weight_fea(y_true, y_pred, weight, all_fea, args)
    # plot_weight_hist(y_true, weight)

    # Calculate average weights for each label in y_true and y_pred
    avg_w = [
        np.mean(weight[y_true == 0]) if np.any(y_true == 0) else 0,
        np.mean(weight[y_true == 1]) if np.any(y_true == 1) else 0,
        np.mean(weight[y_pred == 0]) if np.any(y_pred == 0) else 0,
        np.mean(weight[y_pred == 1]) if np.any(y_pred == 1) else 0
    ]
    
    # return accuracy, bacc, w_f1, f1_macro, f1_micro, confusion, y_true, y_pred
    # return accuracy, w_f1, bacc, f1_macro, cor_strong, cor_acc, avg_w
    return accuracy, w_f1, bacc, f1_macro, avg_w

@torch.no_grad()
def evaltrain2(data, model, classifier, loader, prop_edge_index, args):
    model.eval()
    
    y_true, y_pred, y_pro, all, label, entropy = [], [], [], [], [], []
    # en = (args.en1+args.en2)/2
    for batch in loader:
        idx1, idx2 = batch[0], batch[1]
        idx1, idx2 = idx1.to(args.device), idx2.to(args.device)

        # out = model(data.edge_index, idx1, data.edge_attr, prop_edge_index)
        x = model(prop_edge_index, args)
        edge_emb1 = edge_emb(x, data.edge_index, idx1)
        attr_emb = data.edge_attr[idx1]
        out = classifier(edge_emb1, attr_emb)
        out = torch.softmax(out, dim=1)

        y_true.extend(data.y[idx1].view(-1).cpu().numpy().tolist())
        y_pred.extend(out.argmax(dim=1).cpu().numpy().tolist())
        y_pro.extend(out[:,0].cpu().numpy().tolist())
        idx2=idx2.cpu()
        # all.extend(en[idx2].cpu().numpy().tolist())
        label.extend(args.lab[idx2].cpu().numpy().tolist())
        entropy.extend(args.entropy[idx2].cpu().numpy().tolist())
    return y_true, y_pred, y_pro, all, label, entropy