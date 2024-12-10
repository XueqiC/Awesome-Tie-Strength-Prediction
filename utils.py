import torch
from torch_geometric.data import Data
from collections import defaultdict, Counter
import pandas as pd
from torch.utils.data import Dataset as BaseDataset
import os
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import to_undirected, add_self_loops, degree
import scipy.sparse as sp
from param_parser import parse_args
from torch_geometric.utils import to_networkx
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
from networkx.algorithms.centrality.katz import katz_centrality
from math import comb
import random
import networkx as nx
from model import *
from multiprocessing import Pool

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
def random_split_edge(data, args):
    N=70
    train_size = N
    val_size = N   
    
    label_ratio = args.label_ratio
    num_edges = data.num_edges
    
    perm = torch.randperm(num_edges)
    
    # Split indices
    # train_size = int(num_edges * 0.8)
    # val_size = int(num_edges * 0.1)
    
    train_idx = perm[:int(train_size*label_ratio)]
    val_idx = perm[train_size:train_size + val_size]
    test_idx = perm[train_size + val_size:]

    return {'train': train_idx, 'val': val_idx, 'test': test_idx}

def diff_set(tensor1, tensor2):
    """
    Returns the elements of tensor1 that are not in tensor2
    """
    # Ensure tensor1 and tensor2 are unique
    unique_tensor1 = torch.unique(tensor1)
    unique_tensor2 = torch.unique(tensor2)
    
    # Find elements in tensor1 that are not in tensor2
    mask = ~torch.isin(unique_tensor1, unique_tensor2)
    result = unique_tensor1[mask]
    
    return result

# def random_split_node(data, args):
#     N = 2
#     train_size = N
#     val_size = N
    
#     edge_index = data.edge_index
#     num_nodes = data.num_nodes
#     degrees = torch.zeros(num_nodes, dtype=torch.long)

#     # Calculate node degrees
#     for node in edge_index[0]:
#         degrees[node] += 1
    
#     # Calculate the average degree
#     avg_degree = degrees.float().mean()

#     # Compute deviations from the average degree
#     degree_deviation = torch.abs(degrees - avg_degree)

#     # Convert deviations to probabilities (inverse of deviation)
#     probabilities = 1.0 / (degree_deviation + 1e-8)  # add a small number to avoid division by zero
#     probabilities /= probabilities.sum()  # Normalize to make it a probability distribution

#     balanced = False
#     attempts = 0

#     while not balanced and attempts < 10:  # Limit number of attempts to avoid infinite loop

#         # Randomly select nodes for training and validation using the probabilities
#         all_indices = torch.arange(num_nodes)
#         train_nodes = torch.multinomial(probabilities, train_size, replacement=False)
#         remaining_nodes = torch.tensor([idx for idx in all_indices if idx not in train_nodes])
#         remaining_probs = probabilities[remaining_nodes]

#         val_nodes = torch.multinomial(remaining_probs, val_size, replacement=False)
#         val_nodes = remaining_nodes[val_nodes]  # Map back to original indices

#         # Determine train and validation masks for edges
#         num_edges = edge_index.size(1)
#         train_mask = torch.zeros(num_edges, dtype=torch.bool)
#         val_mask = torch.zeros(num_edges, dtype=torch.bool)

#         for i, (source, target) in enumerate(edge_index.t()):
#             if source.item() in train_nodes or target.item() in train_nodes:
#                 train_mask[i] = True
#             elif source.item() in val_nodes or target.item() in val_nodes:
#                 val_mask[i] = True

#         # Remove training edges from the validation mask
#         val_mask = val_mask & ~train_mask

#         # Check if the number of training and validation edges are similar
#         if abs(train_mask.sum().item() - val_mask.sum().item()) <= 20:  # Allow 5% difference
#             balanced = True
#         attempts += 1

#         # print(f"Attempt {attempts}: Train edges: {train_mask.sum()}, Validation edges: {val_mask.sum()}")

#     # The rest of the edges are test edges
#     test_mask = ~(train_mask | val_mask)
#     train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
#     val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze()
#     test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()

#     return {'train': train_idx, 'val': val_idx, 'test': test_idx}

def random_split_node(data, args):
    N = 4
    train_size = N
    val_size = N
    
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    num_classes = data.y.max().item() + 1  # Assume data.y holds the node class labels
    degrees = torch.zeros(num_nodes, dtype=torch.long)

    # Calculate node degrees
    for node in edge_index[0]:
        degrees[node] += 1
    
    # Calculate class distribution in the neighborhood of each node
    class_distribution = torch.zeros((num_nodes, num_classes), dtype=torch.float)
    
    for source, target in edge_index.t():
        class_distribution[source, data.y[target]] += 1
        class_distribution[target, data.y[source]] += 1

    # Normalize the class distribution for each node, avoid division by zero by adding epsilon to degrees
    class_distribution = class_distribution / (degrees.unsqueeze(1).float() + 1e-8)

    # Compute a "balance score" for each node based on how evenly distributed the classes are
    # Use entropy, ensuring zero values don't affect the log operation
    balance_score = -torch.sum(class_distribution * torch.log(class_distribution + 1e-8), dim=1)

    # Ensure no negative or NaN values in balance_score
    balance_score[balance_score != balance_score] = 0  # Replace NaNs with 0
    balance_score[balance_score < 0] = 0  # Ensure no negative values

    # Normalize the balance scores to be used as probabilities for node selection
    probabilities = balance_score / (balance_score.sum() + 1e-8)  # Normalize probabilities, avoid zero division

    balanced = False
    attempts = 0

    while not balanced and attempts < 10:  # Limit number of attempts to avoid infinite loop

        # Randomly select nodes for training and validation using the probabilities
        all_indices = torch.arange(num_nodes)
        train_nodes = torch.multinomial(probabilities, train_size, replacement=False)
        remaining_nodes = torch.tensor([idx for idx in all_indices if idx not in train_nodes])
        remaining_probs = probabilities[remaining_nodes]

        val_nodes = torch.multinomial(remaining_probs, val_size, replacement=False)
        val_nodes = remaining_nodes[val_nodes]  # Map back to original indices

        # Determine train and validation masks for edges
        num_edges = edge_index.size(1)
        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        val_mask = torch.zeros(num_edges, dtype=torch.bool)

        for i, (source, target) in enumerate(edge_index.t()):
            if source.item() in train_nodes or target.item() in train_nodes:
                train_mask[i] = True
            elif source.item() in val_nodes or target.item() in val_nodes:
                val_mask[i] = True

        # Remove training edges from the validation mask
        val_mask = val_mask & ~train_mask

        # Check if the number of training and validation edges are similar
        if abs(train_mask.sum().item() - val_mask.sum().item()) <= 20:  # Allow 5% difference
            balanced = True
        attempts += 1

    # The rest of the edges are test edges
    test_mask = ~(train_mask | val_mask)
    train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
    val_idx = torch.nonzero(val_mask, as_tuple=False).squeeze()
    test_idx = torch.nonzero(test_mask, as_tuple=False).squeeze()

    return {'train': train_idx, 'val': val_idx, 'test': test_idx}



class EdgeDataset(BaseDataset):
    def __init__(self, idxs):
        self.idxs = idxs

    def __len__(self):
        return self.idxs.shape[0]

    def _get_feed_dict(self, idx):
        return [self.idxs[idx], idx]
    
    def __getitem__(self, idx):
        return self._get_feed_dict(idx)
    
    def collate_fn(self, feed_dicts):
        return torch.tensor([_[0] for _ in feed_dicts]), torch.tensor([_[1] for _ in feed_dicts])
    
def process_edge(edge_index, edge_attr):
    edge_index, edge_attr = to_undirected(edge_index, edge_attr)
    edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, num_nodes = edge_index.max().item() + 1)

    return edge_index, edge_attr

def edge_emb(x, edge_index, edge_idx, method='concat'):
    head, tail = edge_index[:, edge_idx][0], edge_index[:, edge_idx][1]
    head_emb = x[head]
    tail_emb = x[tail]
    
    if method == 'concat':
        edge_emb = torch.cat([head_emb, tail_emb], dim=1)
    elif method == 'hadamard':
        edge_emb = head_emb * tail_emb
    elif method == 'add':
        edge_emb = head_emb + tail_emb
    elif method == 'average':
        edge_emb = (head_emb + tail_emb) / 2
    else:
        raise ValueError("Unsupported method provided. Use 'concat', 'hadamard', 'add', or 'average'.")
    
    return edge_emb


def msg(time_window):
    # Read the file and store edges with timestamps
    edges_with_timestamps = defaultdict(list)
    with open('./dataset/msg.txt', 'r') as file:
        for line in file:
            parts = line.strip().split()
            src, target, timestamp = int(parts[0]), int(parts[1]), int(parts[2])
            edges_with_timestamps[(src, target)].append(timestamp)

    # Calculate the maximum message count within the time window for each edge
    max_counts = {}
    for edge, timestamps in edges_with_timestamps.items():
        timestamps.sort()
        max_count = 0
        for i in range(len(timestamps)):
            window_count = 1
            for j in range(i + 1, len(timestamps)):
                if timestamps[j] - timestamps[i] <= time_window:
                    window_count += 1
                else:
                    break
            max_count = max(max_count, window_count)
        max_counts[edge] = max_count
        
    final_counts = defaultdict(int)
    for (src, target), count in max_counts.items():
        final_counts[(src, target)] = count
        if (target, src) not in final_counts:
            final_counts[(target, src)] = 0
        
    edge_attrs = defaultdict(int)
    for (src, target) in final_counts.keys():
        if (target, src) in edges_with_timestamps:
            timestamps_src_target = sorted(edges_with_timestamps[(src, target)])
            timestamps_target_src = sorted(edges_with_timestamps[(target, src)])
            min_time_gap = float('inf')
            i, j = 0, 0
            while i < len(timestamps_src_target) and j < len(timestamps_target_src):
                time_gap = abs(timestamps_src_target[i] - timestamps_target_src[j])
                if time_gap <= time_window:
                    min_time_gap = time_gap
                    break
                if timestamps_src_target[i] < timestamps_target_src[j]:
                    i += 1
                else:
                    j += 1
            if min_time_gap <= time_window:
                edge_attrs[(src, target)] = 1
                edge_attrs[(target, src)] = 1
            else:
                edge_attrs[(src, target)] = 0
                edge_attrs[(target, src)] = 0
        else:
            edge_attrs[(src, target)] = 0
            edge_attrs[(target, src)] = 0

    # Separate source, target, and counts
    edges = list(max_counts.keys())
    counts = list(max_counts.values())
    attrs = [edge_attrs[edge] for edge in edges]

    # Convert edges and counts to tensors
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    y = torch.tensor(counts, dtype=torch.int)
    edge_attr = torch.tensor(attrs, dtype=torch.float).view(-1,1)
    
    edge_index = remap_edge_index(edge_index)
    num_nodes = len(set(edge_index[0].tolist() + edge_index[1].tolist()))

    # Create the graph data with y as edge feature
    data = Data(edge_index=edge_index, y=y, edge_attr=y.view(-1,1), weight = y, num_nodes=num_nodes)
    # data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)

    return data

def tra(new_data, args):
    edge_index = new_data.edge_index
    y = new_data.y
    num_nodes = new_data.num_nodes

    # Determine strong and weak connections
    strong_mask = y == 1
    weak_mask = ~strong_mask

    # Prepare edge indices and values
    indices_strong = edge_index[:, strong_mask]
    values_strong = torch.ones(strong_mask.sum(), dtype=torch.float)

    indices_weak = edge_index[:, weak_mask]
    values_weak = torch.ones(weak_mask.sum(), dtype=torch.float)

    # Create sparse tensors
    s_s = torch.sparse_coo_tensor(indices_strong, values_strong, (num_nodes, num_nodes))
    s_w = torch.sparse_coo_tensor(indices_weak, values_weak, (num_nodes, num_nodes))

    # Make s_s and s_w symmetric
    s_s = s_s + s_s.t()
    s_s = torch.where(s_s.to_dense() > 0, torch.tensor(1.0), torch.tensor(0.0)).to_sparse()

    s_w = s_w + s_w.t()
    s_w = torch.where(s_w.to_dense() > 0, torch.tensor(1.0), torch.tensor(0.0)).to_sparse()

    compute(s_s, s_w, num_nodes, args)

def compute(s_s, s_w, n, args):
    c0, c1, c2, c4, c5, c6 = [], [], [], [], [], []

    # Convert row slices to dense only when needed, and keep the rest in sparse format
    for i in range(n):
        row_i_s = s_s[i].to_dense()  # Convert only one row to dense
        row_i_w = s_w[i].to_dense()

        # Calculate the number of strong and weak connections for node i
        n_s = torch.sum(row_i_s)
        n_w = torch.sum(row_i_w)

        # Precompute the transpose of the dense row vectors for use in matrix operations
        row_i_s_t = row_i_s.T
        row_i_w_t = row_i_w.T

        if n_s > 1:
            N1 = n_s * (n_s - 1)
            # Perform sparse @ dense multiplication
            p1 = (row_i_s @ s_s @ row_i_s_t) / N1
            p4 = (row_i_s @ s_w @ row_i_s_t) / N1
            c0.append(p1.item())
            c4.append(p4.item())

        if n_w > 1:
            N2 = n_w * (n_w - 1)
            # Perform sparse @ dense multiplication
            p2 = (row_i_w @ s_s @ row_i_w_t) / N2
            p5 = (row_i_w @ s_w @ row_i_w_t) / N2
            c1.append(p2.item())
            c5.append(p5.item())

        if n_s > 0 and n_w > 0:
            N3 = n_s * n_w
            p3 = (row_i_s @ s_s @ row_i_w_t) / N3
            p6 = (row_i_s @ s_w @ row_i_w_t) / N3
            c2.append(p3.item())
            c6.append(p6.item())
            
    plot(np.array(c0),np.array(c1),np.array(c2),np.array(c4),np.array(c5),np.array(c6), args)

    
def plot(c0_np,c1_np,c2_np,c4_np,c5_np,c6_np, args):
# Plotting triadic closure pattern histograms
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # Adjust size as needed

    # Plotting the histograms
    # Note: Adjust the bins, color, and labels as needed
    hist_args = {'bins': 20, 'alpha': 0.75}
    axs[0, 0].hist(c0_np, **hist_args, color='darkolivegreen', label='c0')
    axs[0, 1].hist(c1_np, **hist_args, color='goldenrod', label='c1')
    axs[0, 2].hist(c2_np, **hist_args, color='indigo', label='c2')
    axs[1, 0].hist(c4_np, **hist_args, color='steelblue', label='c4')
    axs[1, 1].hist(c5_np, **hist_args, color='maroon', label='c5')
    axs[1, 2].hist(c6_np, **hist_args, color='navy', label='c6')

    # Customizing each subplot
    for i, ax in enumerate(axs.flat):
        ax.set_xlabel('Probability', fontweight='bold', fontsize=16)
        ax.set_ylabel('Frequency', fontweight='bold', fontsize=16)
        # ax.legend(loc='upper right', fontsize=12, frameon=False)
        ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.rcParams["font.family"] = "Times New Roman"
    if args.dataset == 'ba':
        if args.setting == 1:
            plt.savefig(f'./image/{args.dataset}/setting{args.setting}/stc.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./image/{args.dataset}/setting{args.setting}/stc_{args.thre}.png', dpi=200, transparent=True, bbox_inches='tight')
    elif args.dataset == 'msg':
        if args.setting == 1:
            plt.savefig(f'./image/{args.dataset}/setting{args.setting}/stc_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./image/{args.dataset}/setting{args.setting}/stc_{args.thre}_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')
    elif args.dataset == 'tw':
        plt.savefig(f'./image/{args.dataset}/stc.png', dpi=200, transparent=True, bbox_inches='tight')
    # plt.savefig('./image/msg/msg1_10.png', dpi=200, transparent=True, bbox_inches='tight')
    plt.show()

def remap_edge_index(edge_index): 
    # Find unique nodes and create a mapping from old indices to new indices
    unique_nodes, new_indices = torch.unique(edge_index, return_inverse=True)
    
    # Reshape new_indices to match the shape of edge_index
    remapped_edge_index = new_indices.view(edge_index.size())
    
    return remapped_edge_index

def assign_edge(edge_data, edge_dict, node_neighbor_weights, setting, threshold):
    (src, dst), (weight, label, attr) = edge_data
    result = {'edge_index': [src, dst], 'edge_weight': weight}
    if attr is not None:
        result['edge_attr'] = attr

    # Different settings for label calculation
    if setting == 1:
        is_bidirectional = (dst, src) in edge_dict and (src, dst) in edge_dict
        result['label'] = 1 if is_bidirectional else 0
    elif setting == 2:
        sum_labels = sum(label) if isinstance(label, tuple) else label
        result['label'] = 1 if sum_labels >= threshold else 0
    elif setting == 7:
        is_bidirectional = (dst, src) in edge_dict and (src, dst) in edge_dict
        sum_labels = sum(label) if isinstance(label, tuple) else label
        result['label'] = 1 if is_bidirectional and sum_labels >= threshold else 0
    else:
        i_threshold_index = int(threshold * len(node_neighbor_weights[src]))
        j_threshold_index = int(threshold * len(node_neighbor_weights[dst]))
        i_labels = torch.tensor(np.sort(np.array(node_neighbor_weights[src]))[::-1])
        j_labels = torch.tensor(np.sort(np.array(node_neighbor_weights[dst]))[::-1])
        i_threshold_value = i_labels[max(0, i_threshold_index - 1)]
        j_threshold_value = j_labels[max(0, j_threshold_index - 1)]
        label_ij, label_ji = label

        if setting == 3:
            result['label'] = 1 if label_ij >= i_threshold_value and label_ji >= j_threshold_value else 0
        elif setting == 4:
            result['label'] = 1 if label_ij >= i_threshold_value or label_ji >= j_threshold_value else 0
        elif setting == 5:
            result['label'] = 1 if label_ij + label_ji >= i_threshold_value + j_threshold_value else 0
        elif setting == 6:
            is_bidirectional = (dst, src) in edge_dict and (src, dst) in edge_dict
            result['label'] = 1 if is_bidirectional and label_ij + label_ji >= i_threshold_value + j_threshold_value else 0

    return result

def process(data, setting, threshold):
    edge_index = data.edge_index
    edge_label = data.y
    edge_weight = data.weight if hasattr(data, 'weight') else torch.ones(data.edge_index.shape[1])  # Default to weight 1 if no weight provided
    edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

    # Dictionary for tracking individual directed edge weights, labels, and optional attributes
    edge_dict = {}
    if edge_attr is not None:
        for src, dst, weight, label, attr in zip(edge_index[0], edge_index[1], edge_weight, edge_label, edge_attr):
            edge_dict[(src.item(), dst.item())] = (weight, label, attr)
    else:
        for src, dst, weight, label in zip(edge_index[0], edge_index[1], edge_weight, edge_label):
            edge_dict[(src.item(), dst.item())] = (weight, label, None)

    # Dictionary for undirected edges, merging bidirectional edges into one entry
    undirected_dict = {}
    for (src, dst), (weight, label, attr) in edge_dict.items():
        key = tuple(sorted([src, dst]))
        if key in undirected_dict:
            continue  # Skip if already processed the reverse direction
        reverse_entry = edge_dict.get((dst, src), (0, 0, None if attr is None else torch.zeros_like(attr)))
        reverse_weight, reverse_label, reverse_attr = reverse_entry

        # Combine weights, labels, and attributes appropriately
        combined_weight = weight + reverse_weight
        combined_label = (label, reverse_label) if setting in [4, 5, 6, 7] else label + reverse_label
        combined_attr = (attr + reverse_attr) / 2 if attr is not None else None
        undirected_dict[key] = (combined_weight, combined_label, combined_attr)
        
    if setting in [4, 5, 6, 7]:
        node_neighbor_weights = {i: [] for i in set(edge_index.flatten().tolist())}
        for (src, dst), (_, weights, _) in undirected_dict.items():
            w_ij, w_ji = weights
            # node_neighbor_weights[src].append(w_ij)
            # node_neighbor_weights[dst].append(w_ji)
            node_neighbor_weights[src].append((dst, w_ij))
            node_neighbor_weights[dst].append((src, w_ji))
            
        # node_thresholds = {}
        # for node, weights in node_neighbor_weights.items():
        #     sorted_weights = np.sort(weights)[::-1]
        #     threshold_index = int(threshold * len(sorted_weights)) - 1
        #     node_thresholds[node] = sorted_weights[max(0, threshold_index)]

        # Sort weights and select the top threshold
        temp_labels = {}
        node_thresholds = {}
        for node, neighbors in node_neighbor_weights.items():
            # Sort neighbors based on the weights in descending order
            sorted_neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)
            # Calculate the cutoff index for top threshold percent
            cutoff_index = int(np.ceil(threshold * len(sorted_neighbors)))
            # Assign temporary labels and calculate threshold values
            top_weights = [weight for _, weight in sorted_neighbors[:cutoff_index]]
            node_thresholds[node] = min(top_weights) if top_weights else 0
            for neighbor, weight in sorted_neighbors[:cutoff_index]:
                temp_labels[(node, neighbor)] = 1

    # Prepare lists to gather new graph data
    new_edge_index = []
    new_edge_weights = []
    new_edge_labels = []
    new_edge_attrs = []

    for (src, dst), (weight, label, attr) in undirected_dict.items():
        new_edge_index.append([src, dst])
        new_edge_weights.append(weight)
        if attr is not None:
            new_edge_attrs.append(attr)

        # Different settings for label calculation
        if setting == 1:
            is_bidirectional = (dst, src) in edge_dict and (src, dst) in edge_dict
            new_edge_labels.append(1 if is_bidirectional else 0)
        elif setting == 2:
            sum_labels = sum(label) if isinstance(label, tuple) else label
            new_edge_labels.append(1 if sum_labels >= threshold else 0)
        elif setting == 3:
            is_bidirectional = (dst, src) in edge_dict and (src, dst) in edge_dict
            sum_labels = sum(label) if isinstance(label, tuple) else label
            new_edge_labels.append(1 if is_bidirectional and sum_labels >= threshold else 0)
        else:
            # label_ij, label_ji = label
            # i_threshold_value = node_thresholds[src]
            # j_threshold_value = node_thresholds[dst]
            
            src_dst_label = temp_labels.get((src, dst), 0)
            dst_src_label = temp_labels.get((dst, src), 0)

            if setting == 4:
                # new_edge_labels.append(1 if label_ij >= i_threshold_value and label_ji >= j_threshold_value else 0)
                final_label = 1 if src_dst_label == 1 and dst_src_label == 1 else 0
                new_edge_labels.append(final_label)
            elif setting == 5:
                final_label = 1 if src_dst_label == 1 or dst_src_label == 1 else 0
                new_edge_labels.append(final_label)
                # new_edge_labels.append(1 if label_ij >= i_threshold_value or label_ji >= j_threshold_value else 0)
            elif setting in [6, 7]:
                # new_edge_labels.append(1 if label_ij + label_ji >= i_threshold_value + j_threshold_value else 0)
                i_threshold_value = node_thresholds.get(src, 0)
                j_threshold_value = node_thresholds.get(dst, 0)
                label_ij, label_ji = label
                if setting == 6:
                    # Check if the sum of labels meets or exceeds the sum of thresholds
                    new_edge_labels.append(1 if label_ij + label_ji >= i_threshold_value + j_threshold_value else 0)
                elif setting == 7:
                    is_bidirectional = (dst, src) in edge_dict and (src, dst) in edge_dict
                    new_edge_labels.append(1 if is_bidirectional and label_ij + label_ji >= i_threshold_value + j_threshold_value else 0)

    # Convert lists to tensors for PyG compatibility
    new_edge_index = torch.tensor(new_edge_index).t().contiguous()
    new_edge_weights = torch.tensor(new_edge_weights)
    new_edge_labels = torch.tensor(new_edge_labels)
    new_edge_attrs = torch.stack(new_edge_attrs) if new_edge_attrs else None
    num_nodes = len(set(new_edge_index.flatten().tolist()))

    # Create a new PyG data object with updated edges, weights, labels, and attributes
    new_data = Data(edge_index=new_edge_index, y=new_edge_labels, weight=new_edge_weights, edge_attr=new_edge_attrs, num_nodes=num_nodes)
    return new_data

def build_adjacency_list(edge_index, edge_weights):
    adj_list = defaultdict(set)
    edge_weight_dict = {}
    for idx, (i, j) in enumerate(edge_index.t()):
        # Include all edges in the adjacency list
        adj_list[i.item()].add(j.item())
        adj_list[j.item()].add(i.item())  # Since the graph is undirected
        edge_weight_dict[(i.item(), j.item())] = edge_weights[idx].item()
        edge_weight_dict[(j.item(), i.item())] = edge_weights[idx].item()
    return adj_list, edge_weight_dict

def count_triangles(data):
    """Count triangles of different types based on edge weights."""
    edge_index = data.edge_index
    edge_weights = data.y  # Using 'y' for edge weights

    # Build adjacency list and edge weights dictionary
    adj_list, edge_weights_dict = build_adjacency_list(edge_index, edge_weights)

    triangle_types = {'111': 0, '110': 0, '001': 0, '000': 0}

    # Iterate over each edge and check for triangles
    for i, j in edge_index.t():
        if i < j:  # Ensure each edge is processed only once
            common_neighbors = adj_list[i.item()].intersection(adj_list[j.item()])
            for k in common_neighbors:
                if k > i and k > j:  # Ensure each triangle is counted once
                    edge_sum = edge_weights_dict.get((i.item(), j.item()), 0) + \
                            edge_weights_dict.get((j.item(), k), 0) + \
                            edge_weights_dict.get((i.item(), k), 0)
                    if edge_sum == 3:
                        triangle_types['111'] += 1
                    elif edge_sum == 2:
                        triangle_types['110'] += 1
                    elif edge_sum == 1:
                        triangle_types['001'] += 1
                    elif edge_sum == 0:
                        triangle_types['000'] += 1

    return triangle_types

def plot_triangle_distribution(triangle_types, args):
    """Plot the distribution of triangle types."""
    labels = list(triangle_types.keys())
    counts = list(triangle_types.values())
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.bar(labels, counts, color="navy")
    plt.xlabel('Triangle Type')
    plt.ylabel('Count')
    plt.title('Distribution of Triangle Types')
    plt.xticks(labels, ['All strong', 'One weak', 'Two weak', 'All weak'])
    plt.yscale('log')  # Set the y-axis to logarithmic scale
    if args.dataset == 'ba':
        if args.setting == 1:
            plt.savefig(f'./image/{args.dataset}/setting{args.setting}/triangle_distribution.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./image/{args.dataset}/setting{args.setting}/triangle_distribution_{args.thre}.png', dpi=200, transparent=True, bbox_inches='tight')
    elif args.dataset == 'msg':
        if args.setting == 1:
            plt.savefig(f'./image/{args.dataset}/setting{args.setting}/triangle_distribution_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./image/{args.dataset}/setting{args.setting}/triangle_distribution_{args.thre}_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')
    elif args.dataset == 'tw':
        plt.savefig(f'./image/{args.dataset}/triangle_distribution.png', dpi=200, transparent=True, bbox_inches='tight')
        
    plt.show()
    
def load_data(args):
    dataset = args.dataset
    setting = args.setting
    thre = args.thre
    tw = args.tw
    
    if dataset == 'ba':
        data = pd.read_csv("dataset/bitcoin_alpha.csv")
        n_nodes = len(set(data['source'].tolist() + data['target'].tolist()))
        n_edges = len(data)
        edge_index = torch.tensor([data['source'].tolist(), \
                                data['target'].tolist()], dtype=torch.long)
        weight = torch.tensor(data['rating']).view(-1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder = SentenceTransformer('all-MiniLM-L6-v2', device = device)
        edge_attr = torch.tensor(encoder.encode(data['comment'].fillna('').tolist()))
        d = Data(edge_index = edge_index, edge_attr = edge_attr, \
                num_nodes = n_nodes, num_edges = n_edges, weight = weight)

        positive_edge_indices = d.weight > 0
        subgraph_edge_index = d.edge_index[:, positive_edge_indices]
        subgraph_edge_attr = d.edge_attr[positive_edge_indices, :]
        subgraph_weight = d.weight[positive_edge_indices]
        edge_index = subgraph_edge_index
        edge_index = remap_edge_index(edge_index)
        weights = subgraph_weight
        num_nodes = len(set(edge_index[0].tolist() + edge_index[1].tolist()))
        num_edges = edge_index.size(1)
        sd = Data(edge_index=edge_index, y=weights, edge_attr=subgraph_edge_attr, weight = weights, num_nodes=num_nodes, num_edges=num_edges)
        # sd = Data(edge_index=edge_index, y=weights, num_nodes=num_nodes, num_edges=num_edges)
        
        data = process(sd, setting, thre)
        
    elif dataset == 'msg':
        sd = msg(tw)
        data = process(sd, setting, thre)
        
    elif dataset == 'tw':
        # data = np.load('./dataset/tw/wedge2.npy')
        # data = np.load('./dataset/tw/24_wedge2.npy')
        data = np.load('./dataset/tw/27_wedge2.npy')
        
        # edge_attr = np.load('./dataset/tw/edge_emb.npy')
        # edge_attr = np.load('./dataset/tw/24_edge_emb.npy')
        edge_attr = np.load('./dataset/tw/27_emb.npy')
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        edge_index = torch.tensor(data[:2], dtype=torch.long)
        edge_index = remap_edge_index(edge_index)
        weights = torch.tensor(data[2:], dtype=torch.float).view(-1)  # This is now your y

        # Calculate number of nodes
        num_nodes = len(set(data[0].tolist() + data[1].tolist()))
        # sd = Data(edge_index=edge_index, y=weights, weight = weights, num_nodes=num_nodes)
        sd = Data(edge_index=edge_index, y=weights, weight = weights, edge_attr=edge_attr, num_nodes=num_nodes)

        data = process(sd, setting, thre)
        
    return data

def node_scores(data):
    # Convert PyG data to NetworkX graph (undirected)
    G = to_networkx(data, to_undirected=True)

    # Compute PageRank and Katz centrality
    pr_scores = pagerank(G, personalization={node: 1 for node in G.nodes()}, alpha=0.85)
    katz_scores = katz_centrality(G, alpha=0.005, beta=1.0)

    # Compute degree
    degrees = dict(G.degree())

    # Convert scores to tensors
    node_list = list(G.nodes())
    pr_tensor = torch.tensor([pr_scores[node] for node in node_list])
    katz_tensor = torch.tensor([katz_scores[node] for node in node_list])
    degree_tensor = torch.tensor([degrees[node] for node in node_list])
    tra = node_tra(data) # different types of triangles
    tc_pos = tc(data) # triadic closure

    return pr_tensor, katz_tensor, degree_tensor, tra, tc_pos

def edge_scores(data):
    # Convert PyG data to NetworkX for CN calculation (if not already converted)
    pr_tensor, katz_tensor, degree_tensor, tra, tc_pos = node_scores(data)
    G = to_networkx(data, to_undirected=True)

    # Get edge indices
    edge_index = data.edge_index

    # Compute edge-wise scores vectorized
    pr_edge_scores = (pr_tensor[edge_index[0]] + pr_tensor[edge_index[1]]) / 2
    # katz_edge_scores = (katz_tensor[edge_index[0]] + katz_tensor[edge_index[1]]) / 2
    degree_edge_scores = (degree_tensor[edge_index[0]] + degree_tensor[edge_index[1]]) / 2 #average degree
    edge_tra = (tra[edge_index[0]] + tra[edge_index[1]])/2
    edge_tc = (tc_pos[edge_index[0]] + tc_pos[edge_index[1]])/2
    degree_d = torch.abs(degree_tensor[edge_index[0]] - degree_tensor[edge_index[1]])
    ecc = ec(data) #clustering coefficient

    # Calculate Common Neighbors
    cn_tensor = torch.zeros(edge_index.size(1), dtype=torch.long)
    for i, (u, v) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
        u_neighbors = set(G.neighbors(u))
        v_neighbors = set(G.neighbors(v))
        cn_tensor[i] = len(u_neighbors & v_neighbors)
        
    uo = cn_tensor/(degree_edge_scores*2-2-cn_tensor).view(-1)
    uo = uo.numpy()
    uo = np.nan_to_num(uo, nan=0.0)
    uo = torch.tensor(uo)

    # return pr_edge_scores, katz_edge_scores, degree_edge_scores, cn_tensor, edge_tra, edge_tc
    return pr_edge_scores, degree_edge_scores, edge_tra, edge_tc, degree_d, ecc, cn_tensor, uo
    # return edge_tra, edge_tc, cn_tensor, uo


def edge_cor(data, args):
    # Convert tensors to numpy if they aren't already
    pr_np, katz_np, degree_np, cn_np, tra, _ = edge_scores(data)
    pr_np, katz_np, degree_np, cn_np, tra_np = pr_np.numpy(), katz_np.numpy(), degree_np.numpy(), cn_np.numpy(), tra.numpy()
    labels_np = data.y.numpy()

    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Adjust size as needed
    plt.rcParams["font.family"] = "Times New Roman"

    # Scatter plots
    axs[0, 0].scatter(pr_np, labels_np, alpha=0.2, color='blue', label='PageRank vs Labels')
    axs[0, 1].scatter(katz_np, labels_np, alpha=0.2, color='green', label='Katz vs Labels')
    axs[1, 0].scatter(degree_np, labels_np, alpha=0.2, color='red', label='Degree vs Labels')
    axs[1, 1].scatter(cn_np, labels_np, alpha=0.2, color='purple', label='Common Neighbors vs Labels')

    # Set titles and labels
    axs[0, 0].set_title('PageRank vs Labels')
    axs[0, 1].set_title('Katz vs Labels')
    axs[1, 0].set_title('Degree vs Labels')
    axs[1, 1].set_title('Common Neighbors vs Labels')

    for ax in axs.flat:
        ax.set_xlabel('Metric Value')
        ax.set_ylabel('Labels')
        # ax.legend(loc='upper right')
        
    if args.dataset == 'ba':
        if args.setting == 1:
            plt.savefig(f'./cor_image/{args.dataset}/setting{args.setting}/edge_cor.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./cor_image/{args.dataset}/setting{args.setting}/edge_cor_{args.thre}.png', dpi=200, transparent=True, bbox_inches='tight')
    elif args.dataset == 'msg':
        if args.setting == 1:
            plt.savefig(f'./cor_image/{args.dataset}/setting{args.setting}/edge_cor_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./cor_image/{args.dataset}/setting{args.setting}/edge_cor_{args.thre}_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    
def edge_bar(data, args):
    # Convert tensors to numpy if they aren't already
    pr_np, katz_np, degree_np, cn_np = edge_scores(data)
    pr_np, katz_np, degree_np, cn_np = pr_np.numpy(), katz_np.numpy(), degree_np.numpy(), cn_np.numpy()
    labels_np = data.y.numpy()

    # Calculate the average metrics for each label category
    pr_means = [np.mean(pr_np[labels_np == i]) for i in np.unique(labels_np)]
    katz_means = [np.mean(katz_np[labels_np == i]) for i in np.unique(labels_np)]
    degree_means = [np.mean(degree_np[labels_np == i]) for i in np.unique(labels_np)]
    cn_means = [np.mean(cn_np[labels_np == i]) for i in np.unique(labels_np)]

    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Adjust size as needed
    plt.rcParams["font.family"] = "Times New Roman"

    # Bar plots
    labels = np.unique(labels_np)
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    axs[0, 0].bar(x, pr_means, width, color='blue', label='Avg. PageRank')
    axs[0, 1].bar(x, katz_means, width, color='green', label='Avg. Katz')
    axs[1, 0].bar(x, degree_means, width, color='red', label='Avg. Degree')
    axs[1, 1].bar(x, cn_means, width, color='purple', label='Avg. Common Neighbors')

    # Set titles, labels, and ticks
    axs[0, 0].set_title('Average PageRank by Label')
    axs[0, 1].set_title('Average Katz by Label')
    axs[1, 0].set_title('Average Degree by Label')
    axs[1, 1].set_title('Average Common Neighbors by Label')

    for ax in axs.flat:
        ax.set_xlabel('Label')
        ax.set_ylabel('Average Metric')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        # ax.legend(loc='upper right')
    
    if args.dataset == 'ba':
        if args.setting == 1:
            plt.savefig(f'./cor_image/{args.dataset}/setting{args.setting}/edge_bar.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./cor_image/{args.dataset}/setting{args.setting}/edge_bar_{args.thre}.png', dpi=200, transparent=True, bbox_inches='tight')
    elif args.dataset == 'msg':
        if args.setting == 1:
            plt.savefig(f'./cor_image/{args.dataset}/setting{args.setting}/edge_bar_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./cor_image/{args.dataset}/setting{args.setting}/edge_bar_{args.thre}_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    
def node_tra(data):
    edge_index, edge_labels = data.edge_index, data.y
    
    # Ensure the graph is undirected and has no self-loops
    edge_index, edge_labels = to_undirected(edge_index, edge_labels)

    # Create adjacency list and edge label map
    adjacency_list = defaultdict(set)
    edge_label_map = {}
    for i in range(edge_index.size(1)):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        label = edge_labels[i].item()
        adjacency_list[u].add(v)
        adjacency_list[v].add(u)
        edge_label_map[(min(u, v), max(u, v))] = label

    num_nodes = data.num_nodes
    triangle_types = torch.zeros((num_nodes, 4), dtype=torch.long)

    # Checked pairs to avoid counting triangles multiple times
    checked_pairs = set()

    # Iterate over each node
    for u in adjacency_list:
        for v in adjacency_list[u]:
            if u < v:  # Ensure each pair is processed only once
                for w in adjacency_list[u].intersection(adjacency_list[v]):
                    if u < w and v < w:  # Check if the triangle has not been processed
                        triangle_key = frozenset({u, v, w})
                        if triangle_key not in checked_pairs:
                            checked_pairs.add(triangle_key)
                            # Compute the sum of edge labels
                            label_sum = (
                                edge_label_map[(min(u, v), max(u, v))] +
                                edge_label_map[(min(u, w), max(u, w))] +
                                edge_label_map[(min(v, w), max(v, w))]
                            )
                            # Update the count for each node in the triangle
                            triangle_types[u][label_sum] += 1
                            triangle_types[v][label_sum] += 1
                            triangle_types[w][label_sum] += 1

    return triangle_types

def node_tra_plot(data, args):
    # Convert tensors to numpy for plotting
    pr_np, katz_np, degree_np, triangle_counts_np, _  = node_scores(data)
    pr_np, katz_np, degree_np, triangle_counts_np = pr_np.numpy(), katz_np.numpy(), degree_np.numpy(), triangle_counts_np.numpy()
    
    # Labels for plots
    colors = ['blue', 'green', 'red', 'purple']
    labels = ['All weak', 'One strong', 'Two Strong', 'All strong']
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # 1 row, 3 metrics
    plt.rcParams["font.family"] = "Times New Roman"

    # Scatter plot for each metric against triangle counts, different colors for each triangle type
    for j, metric in enumerate([pr_np, katz_np, degree_np]):
        for i in range(4):
            axs[j].scatter(metric, triangle_counts_np[:, i], alpha=0.2, c=colors[i], label=labels[i])
        axs[j].set_xlabel(f'{["PageRank", "Katz", "Degree"][j]} Value')
        axs[j].set_ylabel('Triangle Count')
        axs[j].set_title(f'{["PageRank", "Katz", "Degree"][j]} vs Triangle Types')
        axs[j].legend()
    
    if args.dataset == 'ba':
        if args.setting == 1:
            plt.savefig(f'./node_cor/{args.dataset}/setting{args.setting}/tra_types.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./node_cor/{args.dataset}/setting{args.setting}/tra_types_{args.thre}.png', dpi=200, transparent=True, bbox_inches='tight')
    elif args.dataset == 'msg':
        if args.setting == 1:
            plt.savefig(f'./node_cor/{args.dataset}/setting{args.setting}/tra_types_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./node_cor/{args.dataset}/setting{args.setting}/tra_types_{args.thre}_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    
def edge_tra_plot(data, args):
    # Convert tensors to numpy for plotting
    pr_np, katz_np, degree_np, cn_np, tra, _ = edge_scores(data)
    pr_np, katz_np, degree_np, cn_np, tra_np = pr_np.numpy(), katz_np.numpy(), degree_np.numpy(), cn_np.numpy(), tra.numpy()
    
    # Labels for plots
    colors = ['blue', 'green', 'red', 'purple']
    labels = ['All weak', 'One strong', 'Two Strong', 'All strong']
    
    # Create a 2x2 subplot layout
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))  # 2 rows, 2 columns
    plt.rcParams["font.family"] = "Times New Roman"
    
    # Metrics list
    metrics = [pr_np, katz_np, degree_np, cn_np]
    metric_names = ["PageRank", "Katz", "Degree", "Common Neighbors"]

    # Scatter plot for each metric against triangle counts, different colors for each triangle type
    for j, (ax, metric) in enumerate(zip(axs.flatten(), metrics)):
        for i in range(4):
            ax.scatter(metric, tra_np[:, i], alpha=0.2, c=colors[i], label=labels[i])
        ax.set_xlabel(f'{metric_names[j]} Value')
        ax.set_ylabel('Triangle Count')
        ax.set_title(f'{metric_names[j]} vs Triangle Types')
        ax.legend()
    
    if args.dataset == 'ba':
        if args.setting == 1:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/tra_types.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/tra_types_{args.thre}.png', dpi=200, transparent=True, bbox_inches='tight')
    elif args.dataset == 'msg':
        if args.setting == 1:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/tra_types_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/tra_types_{args.thre}_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    
def tc(data):
    edge_index, edge_labels = data.edge_index, data.y

    # Convert to undirected and remove self-loops once
    edge_index, edge_labels = to_undirected(edge_index, edge_labels)

    # Efficient mapping from edges to labels
    edge_label_map = {}
    for u, v, l in zip(edge_index[0], edge_index[1], edge_labels):
        u, v = min(u.item(), v.item()), max(u.item(), v.item())
        edge_label_map[(u, v)] = l.item()

    # Create an efficient adjacency structure
    adjacency_list = defaultdict(set)
    for u, v in zip(edge_index[0], edge_index[1]):
        adjacency_list[u.item()].add(v.item())
        adjacency_list[v.item()].add(u.item())

    num_nodes = data.num_nodes
    probabilities_matrix = torch.zeros((num_nodes, 6), dtype=torch.float32)

    # Precompute combinations for possible triangles
    max_degree = max(len(neighbors) for neighbors in adjacency_list.values())
    combination_cache = np.array([comb(n, 2) for n in range(max_degree + 1)])

    # Processing triangles
    for u in adjacency_list:
        neighbors_u = list(adjacency_list[u])
        label_counter = Counter(edge_label_map.get((min(u, v), max(u, v)), -1) for v in neighbors_u)

        # Retrieve cached combinations
        possible_1_1 = combination_cache[label_counter[1]] if label_counter[1] >= 2 else 0
        possible_0_0 = combination_cache[label_counter[0]] if label_counter[0] >= 2 else 0
        possible_1_0 = label_counter[1] * label_counter[0]

        closures = np.zeros(6)

        # Check each pair of neighbors to determine closures
        for i in range(len(neighbors_u)):
            for j in range(i+1, len(neighbors_u)):
                v, w = neighbors_u[i], neighbors_u[j]
                if w in adjacency_list[v]:
                    evu, euw = edge_label_map.get((min(u, v), max(u, v)), -1), edge_label_map.get((min(u, w), max(u, w)), -1)
                    evw = edge_label_map.get((min(v, w), max(v, w)), -1)
                    triangle_type = tuple(([evu, euw, evw]))
                    
                    # Update closures counts
                    if triangle_type == (1, 1, 1):
                        closures[0] += 1
                    elif triangle_type == (1, 1, 0):
                        closures[1] += 1
                    elif triangle_type == (1, 0, 1) or triangle_type == (0, 1, 1):
                        closures[2] += 1
                    elif triangle_type == (1, 0, 0) or triangle_type == (0, 1, 0):
                        closures[3] += 1
                    elif triangle_type == (0, 0, 1):
                        closures[4] += 1
                    elif triangle_type == (0, 0, 0):
                        closures[5] += 1

        # Set probabilities in matrix
        if possible_1_1 > 0:
            probabilities_matrix[u, 0:2] = torch.tensor(closures[0:2]) / possible_1_1
        if possible_1_0 > 0:
            probabilities_matrix[u, 2:4] = torch.tensor(closures[2:4]) / possible_1_0
        if possible_0_0 > 0:
            probabilities_matrix[u, 4:6] = torch.tensor(closures[4:6]) / possible_0_0

    return probabilities_matrix

def node_tc_plot(data, args):
    # Convert tensors to numpy for plotting
    pr_np, katz_np, degree_np, triangle_counts_np, tc_np  = node_scores(data)
    pr_np, katz_np, degree_np, triangle_counts_np, tc_np = pr_np.numpy(), katz_np.numpy(), degree_np.numpy(), triangle_counts_np.numpy(), tc_np
    
    # Labels for plots
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    labels = ['(1,1)-1', '(1,1)-0', '(1,0)-1', '(1,0)-0', '(0,0)-1', '(0,0)-0']

    # Create a 3x3 subplot
    fig, axs = plt.subplots(3, 3, figsize=(18, 18))  # 3 rows, 3 columns
    plt.rcParams["font.family"] = "Times New Roman"

    # Dictionary to map metrics to their respective numpy arrays
    metrics = {'PageRank': pr_np, 'Katz': katz_np, 'Degree': degree_np}

    # Iterate over each metric and plot
    for col, (metric_name, metric_values) in enumerate(metrics.items()):
        for row in range(3):
            for i in range(row * 2, row * 2 + 2):  # Plot two labels per subplot
                if i < len(labels):  # Check to prevent index error on the last pass
                    axs[row, col].scatter(metric_values, tc_np[:, i], alpha=0.2, color=colors[i], label=labels[i])
            axs[row, col].set_xlabel(f'{metric_name} Value')
            axs[row, col].set_ylabel('Triangle Count')
            axs[row, col].set_title(f'{metric_name} vs Triangle Types')
            axs[row, col].legend()

    if args.dataset == 'ba':
        if args.setting == 1:
            plt.savefig(f'./node_cor/{args.dataset}/setting{args.setting}/tc.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./node_cor/{args.dataset}/setting{args.setting}/tc_{args.thre}.png', dpi=200, transparent=True, bbox_inches='tight')
    elif args.dataset == 'msg':
        if args.setting == 1:
            plt.savefig(f'./node_cor/{args.dataset}/setting{args.setting}/tc_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./node_cor/{args.dataset}/setting{args.setting}/tc_{args.thre}_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    
def edge_tc_plot(data, args):
    # Convert tensors to numpy for plotting
    pr_np, katz_np, degree_np, cn_np, tra, tc_np = edge_scores(data)
    pr_np, katz_np, degree_np, cn_np, tra_np, tc = pr_np.numpy(), katz_np.numpy(), degree_np.numpy(), cn_np.numpy(), tra.numpy(), tc_np.numpy()
    
    # Labels for plots
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    labels = ['(1,1)-1', '(1,1)-0', '(1,0)-1', '(1,0)-0', '(0,0)-1', '(0,0)-0']

    # Create a 3x4 subplot layout
    fig, axs = plt.subplots(3, 4, figsize=(24, 18))  # 3 rows, 4 columns
    plt.rcParams["font.family"] = "Times New Roman"

    # Metrics list
    metrics = [pr_np, katz_np, degree_np, cn_np]
    metric_names = ["PageRank", "Katz", "Degree", "Common Neighbors"]

    # Adjusted looping to map each metric across 3 rows and 4 columns
    for col in range(4):  # One column per metric
        metric = metrics[col] if col < len(metrics) else None
        for row in range(3):  # Three rows for pairs of labels
            ax = axs[row, col]
            if metric is not None:
                # Get the label indices for the current row
                label_indices = range(row * 2, row * 2 + 2)
                for label_index in label_indices:
                    if label_index < len(labels):  # Ensure the label index is within the valid range
                        ax.scatter(metric, tc[:, label_index], alpha=0.2, color=colors[label_index], label=labels[label_index])
                ax.set_xlabel(f'{metric_names[col]} Value')
                ax.set_ylabel('Triangle Count')
                ax.set_title(f'{metric_names[col]} vs Triangle Types')
                ax.legend()  # Add legend only for the plotted labels
            else:
                ax.axis('off')  # Turn off axis for empty subplots
    
    if args.dataset == 'ba':
        if args.setting == 1:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/tc.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/tc_{args.thre}.png', dpi=200, transparent=True, bbox_inches='tight')
    elif args.dataset == 'msg':
        if args.setting == 1:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/tc_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/tc_{args.thre}_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    
def ec(data):
    # Convert PyG data to a NetworkX graph
    G = to_networkx(data, to_undirected=True)
    
    # Compute the clustering coefficient for each node
    clustering_coeffs = nx.clustering(G)
    
    # Calculate the sum of clustering coefficients for each edge
    edge_clustering = []
    for edge in data.edge_index.t().tolist():
        # edge[0] is the source node, edge[1] is the target node
        edge_clustering.append(clustering_coeffs[edge[0]] + clustering_coeffs[edge[1]])
    
    # Convert the list to a tensor
    return torch.tensor(edge_clustering)

def un_overlap(data, args):
    _, _, degree_np, cn_np, tra, tc = edge_scores(data)
    uo = cn_np/(degree_np*2-2-cn_np).view(-1)
    uo = uo.numpy()
    uo = np.nan_to_num(uo, nan=0.0)
    return uo
    # edge_labels = data.y
    # edge_labels = np.array(edge_labels)
    # tra = tra.numpy()
    # tc = tc.numpy()
    
    # uo_label(uo, edge_labels, args)
    # uo_tra(uo, tra, args)
    # uo_tc(uo, tc, args)
    
def uo_tc(uo, tc, args):
    
    column_labels = ['(1,1)-1', '(1,1)-0', '(1,0)-1', '(1,0)-0', '(0,0)-1', '(0,0)-0']
    
    # Create a 2x3 grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.rcParams["font.family"] = "Times New Roman"
    
    # Titles for the plots based on column labels
    titles = column_labels  # Direct use of column labels as titles
    
    # First row for columns 1, 3, 5
    first_row_columns = [0, 2, 4]  # 0-indexed columns for 1, 3, 5
    # Second row for columns 2, 4, 6
    second_row_columns = [1, 3, 5]  # 0-indexed columns for 2, 4, 6
    
    # Plotting for the first row
    for idx, col in enumerate(first_row_columns):
        axs[0, idx].scatter(tc[:, col], uo, alpha=0.5, c='blue')
        axs[0, idx].set_title(titles[col])
        # axs[0, idx].set_xlabel(f'TC {titles[col]}')
        axs[0, idx].set_ylabel('Un-Overlap Score (uo)')
        axs[0, idx].grid(True)
    
    # Plotting for the second row
    for idx, col in enumerate(second_row_columns):
        axs[1, idx].scatter(tc[:, col], uo, alpha=0.5, c='red')
        axs[1, idx].set_title(titles[col])
        # axs[1, idx].set_xlabel(f'TC {titles[col]}')
        axs[1, idx].set_ylabel('Un-Overlap Score (uo)')
        # axs[1, idx].grid(True)
    
    if args.dataset == 'ba':
        if args.setting == 1:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/uo_tc.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/uo_tc_{args.thre}.png', dpi=200, transparent=True, bbox_inches='tight')
    elif args.dataset == 'msg':
        if args.setting == 1:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/uo_tc_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/uo_tc_{args.thre}_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')

    plt.show()
    
def uo_tra(uo, tra, args):
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))  # Create a 2x2 grid of subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust space between plots
    plt.rcParams["font.family"] = "Times New Roman"
    
    # Titles for the plots
    titles = ['All weak', 'One strong', 'Two Strong', 'All strong']
    
    for i in range(4):
        ax = axs[i // 2, i % 2]  # Determine the correct subplot position (row, column)
        ax.scatter(tra[:, i], uo, alpha=0.5, c='blue')  # Plot tra[:, i] vs uo
        ax.set_title(titles[i])
        # ax.set_xlabel(f'TRA Column {i + 1}')
        ax.set_ylabel('Un-Overlap Score (uo)')
        # ax.grid(True)
        
    if args.dataset == 'ba':
        if args.setting == 1:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/uo_tra.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/uo_tra_{args.thre}.png', dpi=200, transparent=True, bbox_inches='tight')
    elif args.dataset == 'msg':
        if args.setting == 1:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/uo_tra_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/uo_tra_{args.thre}_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')

    plt.show()

def uo_label(uo, edge_labels, args):
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))  # 1 row, 2 columns
    plt.rcParams["font.family"] = "Times New Roman"

    # Scatter Plot of uo vs edge label
    axs[0].scatter(edge_labels, uo, alpha=0.2, c=['blue' if label == 0 else 'green' for label in edge_labels])
    axs[0].set_xlabel('Edge Label')
    axs[0].set_ylabel('Un-Overlap Score (uo)')

    # Bar Plot to show the average uo for edge label 0 and 1
    avg_uo_label_0 = np.mean(uo[edge_labels == 0])
    avg_uo_label_1 = np.mean(uo[edge_labels == 1])

    labels = ['Weak', 'Strong']
    averages = [avg_uo_label_0, avg_uo_label_1]

    axs[1].bar(labels, averages, color=['blue', 'green'])
    axs[1].set_xlabel('Edge Label')
    axs[1].set_ylabel('Average Un-Overlap Score')
    # axs[1].set_title('Average Un-Overlap Score for Each Edge Label')
    
    if args.dataset == 'ba':
        if args.setting == 1:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/uo_label.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/uo_label_{args.thre}.png', dpi=200, transparent=True, bbox_inches='tight')
    elif args.dataset == 'msg':
        if args.setting == 1:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/uo_label_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')
        else:
            plt.savefig(f'./edge_cor/{args.dataset}/setting{args.setting}/uo_label_{args.thre}_{args.tw:.1e}.png', dpi=200, transparent=True, bbox_inches='tight')

    plt.tight_layout()
    plt.show()
    
def cal_reweight(train_y):
    reweight = torch.tensor([(train_y == _).sum().item() for _ in torch.unique(train_y)])
    reweight = 1/reweight * len(train_y)
    # reweight = torch.tensor([4, 1.3])

    return reweight

def plot_weight_fea(y_true, y_pred, weight, all_fea, args):
    wrong_indices = y_true != y_pred

    wrong_weights = weight[wrong_indices]
    wrong_features = all_fea[wrong_indices]
    wrong_y_true = y_true[wrong_indices]

    # Plot scatter plot of wrongly predicted weight vs feature for each feature dimension
    num_features = wrong_features.shape[1]
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    # plt.rcParams['font.family'] = 'serif'
    # plt.rcParams['font.serif'] = ['Times New Roman']

    for i in range(num_features):
        row = i // 4
        col = i % 4
        axes[row, col].scatter(wrong_features[:, i][wrong_y_true == 0], wrong_weights[wrong_y_true == 0], alpha=0.5, c='red', label='Weak')
        axes[row, col].scatter(wrong_features[:, i][wrong_y_true == 1], wrong_weights[wrong_y_true == 1], alpha=0.5, c='blue', label='Strong')
        # axes[row, col].set_xlabel(f'Feature {i+1}')
        # axes[row, col].set_ylabel('Weight')
        # axes[row, col].set_title(f'Scatter plot of wrongly predicted weight vs Feature {i+1}')
        axes[row, col].legend()

    plt.tight_layout()
    # plt.savefig('image/figure.png')  # Update the path as needed
    plt.savefig(f'image/{args.dataset}/{args.setting}.png')
    plt.show()
    
def plot_weight_hist(y_true, weight):
    fig, ax = plt.subplots(figsize=(10, 6))

    weights_y_true_0 = weight[y_true == 0]
    weights_y_true_1 = weight[y_true == 1]

    ax.hist(weights_y_true_0, bins=30, alpha=0.5, color='red', label='y_true=weak')
    ax.hist(weights_y_true_1, bins=30, alpha=0.5, color='blue', label='y_true=strong')

    ax.set_xlabel('Weight')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Weight Distribution for Wrongly Predicted Instances')
    ax.legend()

    plt.tight_layout()
    plt.savefig('image/figure.png')  # Update the path as needed
    plt.show()
    
def plot_acc_fea(y_true, y_pred, all_fea, args, n_bins=10):

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    for i in range(12):
        ax = axes[i // 4, i % 4]
        ax2 = ax.twinx()  # Create a second y-axis for the histograms
        
        # Calculate bin edges
        bins = np.linspace(all_fea[:, i].min(), all_fea[:, i].max(), n_bins + 1)
        
        # Bin indices for each sample
        bin_indices = np.digitize(all_fea[:, i], bins) - 1
        
        # Accuracy lists for y_true == 0 and y_true == 1
        accuracy_0 = [np.nan] * n_bins  # Initialize with NaNs
        accuracy_1 = [np.nan] * n_bins  # Initialize with NaNs
        counts_0 = [0] * n_bins  # Histogram counts for y_true = 0
        counts_1 = [0] * n_bins  # Histogram counts for y_true = 1
        
        # Calculate accuracy and counts in each bin
        for b in range(n_bins):
            in_bin = bin_indices == b
            if np.sum(in_bin) > 0:
                y_true_bin = y_true[in_bin]
                y_pred_bin = y_pred[in_bin]
                
                # Accuracy for y_true == 0
                if np.any(y_true_bin == 0):
                    accuracy_0[b] = np.mean(y_pred_bin[y_true_bin == 0] == 0)
                counts_0[b] = np.sum(y_true_bin == 0)  # Count for y_true = 0
                
                # Accuracy for y_true == 1
                if np.any(y_true_bin == 1):
                    accuracy_1[b] = np.mean(y_pred_bin[y_true_bin == 1] == 1)
                counts_1[b] = np.sum(y_true_bin == 1)  # Count for y_true = 1
        
        # Plot accuracies
        mid_points = (bins[:-1] + bins[1:]) / 2
        ax.plot(mid_points, accuracy_0, label='Weak', color='navy')
        ax.plot(mid_points, accuracy_1, label='Strong', color='maroon')
        
        # Plot histograms
        ax2.bar(mid_points, counts_0, width=(bins[1]-bins[0])*0.4, align='center', alpha=0.3, color='navy')
        ax2.bar(mid_points + (bins[1]-bins[0])*0.4, counts_1, width=(bins[1]-bins[0])*0.4, align='center', alpha=0.3, color='maroon')
        
        # ax.set_title(f'Dimension {i+1}')
        # ax.set_xlabel('Feature Value Interval')
        # ax.set_ylabel('Accuracy')
        ax.legend(loc='upper left')
        
        # ax2.set_ylabel('Counts')
        # ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'image3/{args.dataset}/{args.setting}.png')
    plt.show()