import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ChebConv, TransformerConv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from torch_geometric.nn import NNConv

class TranConv(nn.Module):
    def __init__(self, n_nodes, n_embed, n_hidden, edge_dim=None):
        super(TranConv, self).__init__()
        
        # Embedding layer for node indices
        self.node_embedding = nn.Embedding(n_nodes, n_embed)
        self.edge_dim = edge_dim
        self.conv1 = TransformerConv(n_embed, n_hidden, edge_dim=edge_dim)
        self.conv2 = TransformerConv(n_hidden, n_hidden)
        
    def reset_parameters(self):
        self.node_embedding.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        
    def forward(self, prop_edge_index, edge_attr, args):
        x = self.node_embedding.weight
        if self.edge_dim is not None:
            x = F.relu(self.conv1(x, prop_edge_index, edge_attr))
        else:
            x = F.relu(self.conv1(x, prop_edge_index))
        x = F.dropout(x, p=args.dropout, training=args.training)
        x = self.conv2(x, prop_edge_index)
        
        return x

    
class GCNEncoder(nn.Module):
    def __init__(self, n_nodes, n_embed, n_hidden):
        super(GCNEncoder, self).__init__()
        
        self.embedding = nn.Embedding(n_nodes, n_embed)
        self.conv1 = GCNConv(n_embed, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_hidden)
    
    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, prop_edge_index, args):
        x, edge_index = self.embedding.weight, prop_edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=args.training)
        x = self.conv2(x, edge_index)
        return x
    
class GATEncoder(nn.Module):
    def __init__(self, n_nodes, n_embed, n_hidden):
        super(GATEncoder, self).__init__()
        
        self.embedding = nn.Embedding(n_nodes, n_embed)
        self.conv1 = GATConv(n_embed, n_hidden)
        self.conv2 = GATConv(n_hidden, n_hidden)
    
    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    
    def forward(self, prop_edge_index, args):
        x, edge_index = self.embedding.weight, prop_edge_index
        x = F.dropout(x, p=args.dropout, training=args.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=args.training)
        x = self.conv2(x, edge_index)
        return x
    
class SAGEEncoder(nn.Module):
    def __init__(self, n_nodes, n_embed, n_hidden):
        super(SAGEEncoder, self).__init__()
        
        self.embedding = nn.Embedding(n_nodes, n_embed)
        self.conv1 = SAGEConv(n_embed, n_hidden)
        self.conv2 = SAGEConv(n_hidden, n_hidden)
    
    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, prop_edge_index, args):
        x, edge_index = self.embedding.weight, prop_edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=args.training)
        x = self.conv2(x, edge_index)
        return x
    
class ChebEncoder(nn.Module):
    def __init__(self, n_nodes, n_embed, n_hidden):
        super(ChebEncoder, self).__init__()
        
        self.embedding = nn.Embedding(n_nodes, n_embed)
        self.conv1 = ChebConv(n_embed, n_hidden, K = 2)
        self.conv2 = ChebConv(n_hidden, n_hidden, K = 2)
    
    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.conv1.reset_parameters()
        
    def forward(self, prop_edge_index, args):
        x = self.conv1(self.embedding.weight, prop_edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=args.dropout, training=args.training)
        x = self.conv2(x, prop_edge_index)
        return x
    
class mlp(nn.Module):
    def __init__(self, n_nodes, n_embed, n_hidden):
        super(mlp, self).__init__()
        self.embedding = nn.Embedding(n_nodes, n_embed)
        self.fc1 = nn.Linear(n_embed, n_hidden)
    
    def reset_parameters(self):
        self.embedding.reset_parameters()
        self.fc1.reset_parameters()
    
    def forward(self, prop_edge_index, args):
        x = self.embedding.weight
        x = F.softmax(self.fc1(x), dim=1)
        return x

class Classifier(nn.Module):
    def __init__(self, n_edge_attr, n_out, args):
        super(Classifier, self).__init__()
        n_hidden = args.n_hidden
        dropout = args.dropout
        self.edge_attr_lin = nn.Linear(n_edge_attr, n_hidden)
        self.relu = nn.ReLU()
        if args.method == 'concat':
            self.classifier_1 = nn.Linear(n_hidden*2 + n_hidden, n_hidden)
        else: 
            self.classifier_1 = nn.Linear(n_hidden*2, n_hidden)
        self.classifier_2 = nn.Linear(n_hidden, n_out)
        self.dropout = nn.Dropout(dropout)
    
    def reset_parameters(self):
        self.edge_attr_lin.reset_parameters()
        self.classifier_1.reset_parameters()
        self.classifier_2.reset_parameters()
        
    def forward(self, emb, attr):
        edge_attr_emb = self.relu(self.edge_attr_lin(attr))
        edge_emb = torch.cat([emb, edge_attr_emb], dim=1)
        edge_emb = self.dropout(F.relu(self.classifier_1(edge_emb)))

        return F.softmax(self.classifier_2(edge_emb), dim=1)
    
class Classifier2(nn.Module): # no edge attribute
    def __init__(self, n_out, args):
        super(Classifier2, self).__init__()
        n_hidden = args.n_hidden
        dropout = args.dropout
        if args.method == 'concat':
            self.classifier_1 = nn.Linear(n_hidden*2, n_hidden)
        else: 
            self.classifier_1 = nn.Linear(n_hidden, n_hidden)
        self.relu = nn.ReLU()
        self.classifier_2 = nn.Linear(n_hidden, n_out)
        self.dropout = nn.Dropout(dropout)
    
    def reset_parameters(self):
        self.classifier_1.reset_parameters()
        self.classifier_2.reset_parameters()
        
    def forward(self, emb):
        edge_emb = self.dropout(F.relu(self.classifier_1(emb)))
        
        return F.softmax(self.classifier_2(edge_emb), dim=1)
    
def rf(data, edge_idxs): #Random Forest Herustic method for feature importance
        train_idx, val_idx, test_idx = edge_idxs['train'], edge_idxs['val'], edge_idxs['test']
        features = data.all_fea
        labels = data.y
        weight = data.weight
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(features[train_idx], labels[train_idx])
        predictions_test = clf.predict(features[test_idx])
        weight = np.array(weight[test_idx])
        feature_importances = clf.feature_importances_
        
        features_test = features[test_idx,:]
        correlations_test = np.zeros(features_test.shape[1])
        for i in range(features_test.shape[1]):
            feature = features_test[:, i]
            correlation = np.corrcoef(feature, predictions_test)[0, 1]
            correlations_test[i] = correlation
        
        true = np.array(labels[test_idx])
        correlations_cor = np.zeros(features_test.shape[1])
        for i in range(features_test.shape[1]):
            feature = features_test[:, i]
            # Safe correlation calculation: np.corrcoef might fail if there is no variance in 'feature' or 'correct_predictions'
            if np.std(feature) == 0 or np.std(true) == 0:
                correlation = 0
            else:
                correlation = np.corrcoef(feature, true)[0, 1]
            correlations_cor[i] = correlation
            
        test_acc = accuracy_score(labels[test_idx], predictions_test)
        f1 = f1_score(labels[test_idx], predictions_test, average='binary')
        bacc = balanced_accuracy_score(labels[test_idx], predictions_test)
        f1_macro = f1_score(labels[test_idx], predictions_test, average='macro')

        avg_w = [
            np.mean(weight[true == 0]) if np.any(true == 0) else 0,
            np.mean(weight[true == 1]) if np.any(true == 1) else 0,
            np.mean(weight[predictions_test == 0]) if np.any(predictions_test == 0) else 0,
            np.mean(weight[predictions_test == 1]) if np.any(predictions_test == 1) else 0
        ]
        
        return test_acc, f1, bacc, f1_macro, correlations_test, correlations_cor, avg_w