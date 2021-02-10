import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import sys
import torch
import torch.nn as nn
import scipy.io as sio
import os
import random


def init_seeds(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data_dblp(args):
    dataset = args.dataset
    metapaths = args.metapaths_list
    sc = args.sc

    if dataset == 'acm':
        data = sio.loadmat('data/{}.mat'.format(dataset))
    else:
        data = pkl.load(open('data/{}.pkl'.format(dataset), "rb"))
    label = data['label']
    N = label.shape[0]

    truefeatures = data['feature'].astype(float)
    rownetworks = [data[metapath] + np.eye(N) * sc for metapath in metapaths]

    rownetworks = [sp.csr_matrix(rownetwork) for rownetwork in rownetworks]

    truefeatures = sp.lil_matrix(truefeatures)

    idx_train = data['train_idx'].ravel()
    idx_val = data['val_idx'].ravel()
    idx_test = data['test_idx'].ravel()

    truefeatures_list = []
    for _ in range(len(rownetworks)):
        truefeatures_list.append(truefeatures)

    return rownetworks, truefeatures_list, label, idx_train, idx_val, idx_test


def parse_skipgram(fname):
    with open(fname) as f:
        toks = list(f.read().split())
    nb_nodes = int(toks[0])
    nb_features = int(toks[1])
    ret = np.empty((nb_nodes, nb_features))
    it = 2
    for i in range(nb_nodes):
        cur_nd = int(toks[it]) - 1
        it += 1
        for j in range(nb_features):
            cur_ft = float(toks[it])
            ret[cur_nd][j] = cur_ft
            it += 1
    return ret


# Process a (subset of) a TU dataset into standard form
def process_tu(data, nb_nodes):
    nb_graphs = len(data)
    ft_size = data.num_features

    features = np.zeros((nb_graphs, nb_nodes, ft_size))
    adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    labels = np.zeros(nb_graphs)
    sizes = np.zeros(nb_graphs, dtype=np.int32)
    masks = np.zeros((nb_graphs, nb_nodes))

    for g in range(nb_graphs):
        sizes[g] = data[g].x.shape[0]
        features[g, :sizes[g]] = data[g].x
        labels[g] = data[g].y[0]
        masks[g, :sizes[g]] = 1.0
        e_ind = data[g].edge_index
        coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(nb_nodes, nb_nodes))
        adjacency[g] = coo.todense()

    return features, adjacency, labels, sizes, masks


def micro_f1(logits, labels):
    # Compute predictions
    preds = torch.round(nn.Sigmoid()(logits))

    # Cast to avoid trouble
    preds = preds.long()
    labels = labels.long()

    # Count true positives, true negatives, false positives, false negatives
    tp = torch.nonzero(preds * labels).shape[0] * 1.0
    tn = torch.nonzero((preds - 1) * (labels - 1)).shape[0] * 1.0
    fp = torch.nonzero(preds * (labels - 1)).shape[0] * 1.0
    fn = torch.nonzero((preds - 1) * labels).shape[0] * 1.0

    # Compute micro-f1 score
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


"""
 Prepare adjacency matrix by expanding up to a given neighbourhood.
 This will insert loops on every node.
 Finally, the matrix is converted to bias vectors.
 Expected shape: [graph, nodes, nodes]
"""


def adj_to_bias(adj, sizes, nhood=1):
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


###############################################
# This section of code adapted from tkipf/gcn #
###############################################

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):  # {'pubmed', 'citeseer', 'cora'}
    """Load data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + 500)

    return adj, features, labels, idx_train, idx_val, idx_test


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation"""
    # standardize data
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def process_adj_gat(adj):
    # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # Tricky implementation of official GAT
    adj = (adj + sp.eye(adj.shape[0])).todense()
    for x in range(0, adj.shape[0]):
        for y in range(0, adj.shape[1]):
            if adj[x, y] == 0:
                adj[x, y] = -9e15
            elif adj[x, y] >= 1:
                adj[x, y] = 0
            else:
                print(adj[x, y], 'error')
    adj = torch.FloatTensor(np.array(adj))
    # adj = sp.coo_matrix(adj)
    return adj


from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor, fill_diag, sum, mul
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops


def adj_to_edge_idx(adj):
    edge_index = adj._indices()
    edge_weight = adj._values()
    return edge_index, edge_weight


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):
    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def dropout_adj(adj, percent, normalization=None):
    """
    Randomly drop edge and preserve percent% edges.
    """
    nnz = adj._nnz()
    if percent > 0:
        perm = np.random.permutation(nnz)
        preserve_nnz = int(nnz * (1 - percent))
        perm = perm[:preserve_nnz]
        pre_indices = adj._indices()[:, perm]
        pre_val = adj._values()[perm]
        pre_indices, pre_val = gcn_norm(edge_index=pre_indices, edge_weight=pre_val)
        if adj.get_device() < 0:
            pre_adj = torch.sparse_coo_tensor(indices=pre_indices, values=pre_val, size=adj.size())
        else:
            pre_adj = torch.sparse_coo_tensor(indices=pre_indices, values=pre_val, size=adj.size(),
                                              device=adj.get_device())
    else:
        pre_adj = adj

    return pre_adj


def drop_feature(x, drop_prob):
    """
    Randomly drop node features.
    """
    if drop_prob == 0:
        return x
    else:
        if torch.is_tensor(x):
            drop_mask = torch.empty(
                (x.size(-1),),
                dtype=torch.float32,
                device=x.device).uniform_(0, 1) < drop_prob
            res = x.clone()
            res[:, :, drop_mask] = 0
        else:
            res = []
            for _x in x:
                drop_mask = torch.empty(
                    (_x.size(-1),),
                    dtype=torch.float32,
                    device=_x.device).uniform_(0, 1) < drop_prob
                _x = _x.clone()
                _x[:, :, drop_mask] = 0
                res.append(_x)
        return res


def get_dir_name(models_dir):
    """Gets a directory to save the model.

    If the directory already exists, then append a new integer to the end of
    it. This method is useful so that we don't overwrite existing models
    when launching new jobs.

    Args:
        models_dir: The directory where all the models are.

    Returns:
        The name of a new directory to save the training logs and model weights.
    """
    if not os.path.exists(models_dir):
        save_dir = os.path.join(models_dir, '0')
        os.makedirs(save_dir)
    else:
        existing_dirs = np.array(
            [
                d
                for d in os.listdir(models_dir)
                if os.path.isdir(os.path.join(models_dir, d))
            ]
        ).astype(np.int)
        if len(existing_dirs) > 0:
            dir_id = str(existing_dirs.max() + 1)
        else:
            dir_id = "1"
        save_dir = os.path.join(models_dir, dir_id)
        os.makedirs(save_dir)
    return save_dir


def get_num_edges(adjs):
    num_edges = []
    for adj in adjs:
        num_edges.append(adj._indices().shape[1])
    # min_num_edges = np.min(num_edges)
    sum = np.sum(num_edges)
    ratio_edges = [num_edges[i] / sum for i in range(len(adjs))]
    return num_edges, ratio_edges, sum


def get_adj_idx(adjs, dropout):
    res = []
    for adj in adjs:
        num_edges = adj._indices().shape[1] * (1 - dropout)
        adj_idx = np.arange(num_edges)
        res.append(adj_idx)
    return res


def get_adj_batch(adj_idx, batch_size):
    batch_idxs = []
    for idx in adj_idx:
        np.random.shuffle(idx)
        batch_idxs.append(idx[0:batch_size])
    return batch_idxs


def structured_node_negative_sampling(idx_1, edge_index_i, neg_sample_ratio, num_nodes=None):
    i = torch.cat([edge_index_i for _ in range(neg_sample_ratio)], dim=-1)
    i = i.to('cpu')
    k = torch.randint(num_nodes, (i.size(0),), dtype=torch.long)
    idx_2 = i * num_nodes + k

    mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
    rest = mask.nonzero(as_tuple=False).view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.randint(num_nodes, (rest.numel(),), dtype=torch.long)
        idx_2 = i[rest] * num_nodes + tmp
        mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
        k[rest] = tmp
        rest = rest[mask.nonzero(as_tuple=False).view(-1)]
    return k.to(edge_index_i.device)


def one_one_negative_sampling(adjs, batchs=None, neg_sample_ratio=1, num_nodes=None):
    # num_edges = adj._indices().shape[1]
    # idx = np.random.permutation(num_edges)[:sample_size]
    # pos_indices = adj._indices()[:, idx]

    num_nodes = maybe_num_nodes(adjs[0]._indices(), num_nodes)
    num_rels = len(adjs)
    idx_1 = []
    for _r in range(num_rels):
        i, j = adjs[_r]._indices().to('cpu')
        idx = i * num_nodes + j
        idx_1.append(idx)
    idx_1 = torch.cat(idx_1, dim=0)

    i = []
    j = []
    ks = []
    for r in range(len(adjs)):
        adj_i = adjs[r]._indices()[0]
        adj_j = adjs[r]._indices()[1]
        if batchs is not None:
            adj_i = adj_i[batchs[r]]
            adj_j = adj_j[batchs[r]]
        k = structured_node_negative_sampling(idx_1, adj_i, neg_sample_ratio, num_nodes).view(adj_i.size(0), -1)
        i.append(adj_i)
        j.append(adj_j)
        ks.append(k)
    return i, j, ks


def structured_relation_negative_sampling(idx_1, d, r, adj, num_rels, num_nodes, neg_sample_ratio=1):
    adj = torch.cat([adj for _ in range(neg_sample_ratio)], dim=-1)
    i, j = adj[0].to('cpu'), adj[1].to('cpu')
    idx_2 = i * num_nodes + j
    other_idx = torch.cat([idx_1[_r] for _r in range(num_rels) if _r != r], dim=-1)

    neg_r_base = list(np.arange(num_rels))
    neg_r_base.remove(r)

    k = torch.tensor(np.random.choice(neg_r_base, size=i.size(0)))
    mask = torch.from_numpy(np.isin(idx_2, other_idx)).to(torch.bool)  # mask for replicate idx
    rest = mask.nonzero(as_tuple=False).view(-1)  # idx of mask
    flag_break = 0
    while rest.numel() > 0:
        pre_tmp = rest.numel()
        for _i in range(rest.numel()):
            key = idx_2[rest[_i]]  # edge_idx
            neg_r = k[rest[_i]]  # negative relation
            if key.item() in d[neg_r]:
                continue
            else:
                mask[rest[_i]] = False
        rest = mask.nonzero(as_tuple=False).view(-1)
        tmp = torch.tensor(np.random.choice(neg_r_base, size=(rest.numel(),)))
        k[rest] = tmp
        if len(tmp) == pre_tmp:
            flag_break += 1
            if flag_break > num_rels - 1:
                k = k[~mask]
                break

    return k.to(adj.device), ~mask


def one_one_rel_negative_sampling(adjs, batchs, neg_sample_ratio=1, num_nodes=None):
    # change sparse coo_matrix to edge_index
    # num_rels = len(adjs)
    # edge_indexs = []
    # for sub_g in range(num_rels):
    #     adj = adjs[sub_g]
    #     indices = adj._indices()
    #     edge_index = indices.clone()
    #     edge_indexs.append(edge_index)
    # rest_edge_indexs = rel_negative_sampling(edge_indexs)
    # rel_pos_edge_indexs = []
    # for r in range(num_rels):
    #     idx = np.random.permutation(rest_edge_indexs[r].shape[-1])[:sample_size]
    #     rel_pos_edge_index = rest_edge_indexs[r][:, idx]
    #     rel_pos_edge_indexs.append(rel_pos_edge_index)
    num_nodes = maybe_num_nodes(adjs[0]._indices(), num_nodes)
    num_rels = len(adjs)

    idx_1 = []
    d = []
    for r in range(num_rels):
        i, j = adjs[r]._indices().to('cpu')
        idx = i * num_nodes + j
        idx_1.append(idx)
        d.append(dict(zip(idx.tolist(), [r for _ in range(len(idx))])))

    i = []
    j = []
    rs = []
    for _r in range(num_rels):
        adj = adjs[_r]._indices()
        if batchs is not None:
            adj = adj[:, batchs[_r]]
        r, mask = structured_relation_negative_sampling(idx_1, d, _r, adj, num_rels, num_nodes)
        i.append(adj[0][mask])
        j.append(adj[1][mask])
        rs.append(r)
    return i, j, rs
