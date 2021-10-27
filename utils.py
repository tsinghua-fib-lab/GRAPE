import numpy as np
import scipy.sparse as sp
import h5py
import torch

# Load in amazon dataset
def load_data_amazon():
    with h5py.File('../data/amazon/amazon.hdf5', 'r') as f:
        adj = f['adj'].value
        features = f['feats'].value
        labels = f['label'].value
        idx_train = f['idx_train'].value
        idx_val = f['idx_val'].value
        idx_test = f['idx_test'].value

    features = sp.lil_matrix(features)
    adj = sp.csr_matrix(adj)
    
    features = row_normalize(features)
    features = np.array(features.todense())

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test) 

    return adj, features, labels, idx_train, idx_val, idx_test

# Load in social datasets
def load_data_social(dataset_str):
    with h5py.File('../data/social/'+dataset_str + '.hdf5', 'r') as f:
        adj = f['adj'].value
        features = f['feats'].value
        labels = f['label'].value
        idx_train = f['idx_train'].value
        idx_val = f['idx_val'].value
        idx_test = f['idx_test'].value

    features = sp.lil_matrix(features)
    adj = sp.csr_matrix(adj)
    
    features = row_normalize(features)
    features = np.array(features.todense())

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test) 

    return adj, features, labels, idx_train, idx_val, idx_test

# Load in citation datasets
def load_data_cite(dataset_str):
    with h5py.File('../data/cite/'+dataset_str + '.hdf5', 'r') as f:
        adj = f['adj'].value
        features = f['feats'].value
        labels = f['label'].value
        idx_train = f['idx_train'].value
        idx_val = f['idx_val'].value
        idx_test = f['idx_test'].value

    features = sp.lil_matrix(features)
    adj = sp.csr_matrix(adj)

    features = row_normalize(features)
    features = np.array(features.todense())

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1])
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test) 

    return adj, features, labels, idx_train, idx_val, idx_test

# Row-normalize sparse matrix
def row_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

# Normalize sparse matrix
def normalize(mx):
    rowsum = np.array(mx.sum(1)) * 0.5
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    colsum = np.array(mx.sum(0)) * 0.5
    c_inv = np.power(colsum, -1).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    c_mat_inv = sp.diags(c_inv)
    mx = r_mat_inv.dot(mx)
    mx = mx.dot(c_mat_inv)
    return mx

# Compute classification accuracy
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# Convert a scipy sparse matrix to a torch sparse tensor
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# Pool of undirected motifs
motif_line = np.zeros((2, 2), dtype=np.int32)
motif_line[0, 1] = 1
motif_line[1, 0] = 1
motif_line_gene = [None, motif_line]

motif_twoline = np.zeros((3, 3), dtype=np.int32)
motif_twoline[0, 1] = 1
motif_twoline[1, 0] = 1
motif_twoline[1, 2] = 1
motif_twoline[2, 1] = 1
motif_twoline_gene = [motif_line, motif_twoline]

motif_twostar = np.zeros((3, 3), dtype=np.int32)
motif_twostar[0, 1] = 1
motif_twostar[1, 0] = 1
motif_twostar[0, 2] = 1
motif_twostar[2, 0] = 1
motif_twostar_gene = [motif_line, motif_twostar]

motif_triangle = np.zeros((3, 3), dtype=np.int32)
motif_triangle[0, 1] = 1
motif_triangle[0, 2] = 1
motif_triangle[1, 2] = 1
motif_triangle[1, 0] = 1
motif_triangle[2, 0] = 1
motif_triangle[2, 1] = 1
motif_triangle_gene = [motif_twoline, motif_triangle]

motif_trianglestar = np.zeros((4, 4), dtype=np.int32)
motif_trianglestar[0, 1] = 1
motif_trianglestar[0, 2] = 1
motif_trianglestar[1, 2] = 1
motif_trianglestar[1, 0] = 1
motif_trianglestar[2, 0] = 1
motif_trianglestar[2, 1] = 1
motif_trianglestar[0, 3] = 1
motif_trianglestar[3, 0] = 1
motif_trianglestar_gene = [motif_triangle, motif_trianglestar]

motif_threestar = np.zeros((4, 4), dtype=np.int32)
motif_threestar[0, 1] = 1
motif_threestar[1, 0] = 1
motif_threestar[0, 2] = 1
motif_threestar[2, 0] = 1
motif_threestar[0, 3] = 1
motif_threestar[3, 0] = 1
motif_threestar_gene = [motif_twostar, motif_threestar]

motif_threeline = np.zeros((4, 4), dtype=np.int32)
motif_threeline[0, 1] = 1
motif_threeline[1, 0] = 1
motif_threeline[1, 2] = 1
motif_threeline[2, 1] = 1
motif_threeline[2, 3] = 1
motif_threeline[3, 2] = 1
motif_threeline_gene = [motif_twoline, motif_threeline]

motif_rectangle = np.zeros((4, 4), dtype=np.int32)
motif_rectangle[0, 1] = 1
motif_rectangle[1, 0] = 1
motif_rectangle[1, 2] = 1
motif_rectangle[2, 1] = 1
motif_rectangle[2, 3] = 1
motif_rectangle[3, 2] = 1
motif_rectangle[0, 3] = 1
motif_rectangle[3, 0] = 1
motif_rectangle_gene = [motif_threeline, motif_rectangle]

motif_semifourclique = np.zeros((4, 4), dtype=np.int32)
motif_semifourclique[0, 1] = 1
motif_semifourclique[1, 0] = 1
motif_semifourclique[0, 2] = 1
motif_semifourclique[2, 0] = 1
motif_semifourclique[0, 3] = 1
motif_semifourclique[3, 0] = 1
motif_semifourclique[1, 2] = 1
motif_semifourclique[2, 1] = 1
motif_semifourclique[2, 3] = 1
motif_semifourclique[3, 2] = 1
motif_semifourclique_gene = [motif_rectangle, motif_semifourclique]

motif_fourclique = np.zeros((4, 4), dtype=np.int32)
motif_fourclique[0, 1] = 1
motif_fourclique[1, 0] = 1
motif_fourclique[0, 2] = 1
motif_fourclique[2, 0] = 1
motif_fourclique[0, 3] = 1
motif_fourclique[3, 0] = 1
motif_fourclique[1, 2] = 1
motif_fourclique[2, 1] = 1
motif_fourclique[1, 3] = 1
motif_fourclique[3, 1] = 1
motif_fourclique[2, 3] = 1
motif_fourclique[3, 2] = 1
motif_fourclique_gene = [motif_semifourclique,motif_fourclique]

# Pool of directed motifs
motif_from = np.zeros((2, 2), dtype=np.int32)
motif_from[0, 1] = 1
motif_from_gene = [None, motif_from]

motif_fromto = np.zeros((3, 3), dtype=np.int32)
motif_fromto[0, 1] = 1
motif_fromto[2, 0] = 1
motif_fromto_gene = [motif_from, motif_fromto]

motif_to = np.zeros((2, 2), dtype=np.int32)
motif_to[1, 0] = 1
motif_to_gene = [None, motif_to]

motif_bi_gene = [motif_to, motif_line]

motif_oneto = np.zeros((3, 3), dtype=np.int32)
motif_oneto[1, 0] = 1
motif_oneto[0, 1] = 1
motif_oneto[1, 2] = 1
motif_oneto_gene = [motif_line, motif_oneto]

motif_twobi_gene = [motif_oneto, motif_twoline]

motif_twoback = np.zeros((3, 3), dtype=np.int32)
motif_twoback[1, 0] = 1
motif_twoback[0, 1] = 1
motif_twoback[1, 2] = 1
motif_twoback[2, 1] = 1
motif_twoback[2, 0] = 1
motif_twoback_gene = [motif_twoline, motif_twoback]

motif_threebi_gene = [motif_twoback, motif_triangle]

def read_data(data):
    if(data.startswith('cite')):
        minor_data = data.split('|')[1]
        adj, features, labels, idx_train, idx_val, idx_test = load_data_cite(minor_data)
        flag_direct = False
        population_test = [motif_line_gene, motif_twoline_gene, motif_threeline_gene, motif_triangle_gene, motif_trianglestar_gene]
        select_index = [0,1,2,3,4]
    elif(data.startswith('social')):
        minor_data = data.split('|')[1]
        adj, features, labels, idx_train, idx_val, idx_test = load_data_social(minor_data)
        flag_direct = False
        population_test = [motif_line_gene, motif_twoline_gene,  motif_triangle_gene, motif_trianglestar_gene, motif_threeline_gene, motif_rectangle_gene, motif_semifourclique_gene, motif_fourclique_gene]
        select_index = [0,1,2,3,7]
    elif(data.startswith('amazon')):
        adj, features, labels, idx_train, idx_val, idx_test = load_data_amazon()
        flag_direct = True
        population_test = [motif_from_gene, motif_to_gene, motif_fromto_gene, motif_bi_gene, motif_oneto_gene, motif_twobi_gene, motif_twoback_gene, motif_threebi_gene]
        select_index = [0,1,2,3,7]
    return adj, features, labels, idx_train, idx_val, idx_test, flag_direct, population_test, select_index