from __future__ import division
from __future__ import print_function

import time
from motif_search import *
from utils import *
from models import GRAPE
import setproctitle
import os
import scipy.sparse as sp
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import search
import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    # 'cite|cora', 'cite|citeseer', 'amazon', 'social|Amherst', 'social|Hamilton', 'social|Rochester', 'social|Lehigh', 'social|Johns Hopkins'
    parser.add_argument('--data', default='cite|cora') 
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--lr', default=0.003)
    parser.add_argument('--wd', default=0.00003)
    parser.add_argument('--dropout', default=0.5)
    parser.add_argument('--hid', default=32)
    return parser

parser = get_parser()
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
attn = True     # Switch for squeeze-and-excite net
flag_acc = True # Accumulate motif count or not
model_name = 'GRAPE'
setproctitle.setproctitle(model_name)


# set random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# grape model hyperparameter setting
num_genes = 5
early_stopping = 50
nepoch = 500
nlayer = 2
test_run = 10


# compute accuracy and loss of the trained models
def evaluate(pred, target, idx):
    pred = F.log_softmax(pred, dim=1)
    loss = F.nll_loss(pred[idx], target[idx])
    acc = accuracy(pred[idx], target[idx]).item() 
    return loss, acc

def train_model(nlayer, nepoch, candidate_adj, features, labels, idx_train, idx_val, idx_test, attn, lr, weight_decay, dropout, hidden):
    # flatten the ADJ of different motifs and add in a self-loop
    ngene = len(candidate_adj)
    nrole = [len(item) for item in candidate_adj]
    nclass = labels.max().item() + 1
    model = GRAPE(nfeat=features.shape[1], nlayer=nlayer, nhid=hidden, nclass=nclass, nrole=nrole, ngene=ngene, dropout=dropout, attn=attn)
    cur_lr = lr
    optimizer = optim.Adam(model.parameters(), lr=cur_lr, weight_decay=weight_decay)

    if torch.cuda.is_available():
        model.cuda()
        features = features.cuda()
        candidate_adj = [[itemtemp.cuda() for itemtemp in temp] for temp in candidate_adj]
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    loss_val_list = []
    # Train model
    t_total = time.time()
    for epoch in range(nepoch):
        # Construct feed dictionary
        model.train()
        optimizer.zero_grad()
        output = model(features, candidate_adj)
        
        loss_train, acc_train = evaluate(output, labels, idx_train)

        loss_train.backward()
        optimizer.step()

        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, candidate_adj)

        loss_val, acc_val = evaluate(output, labels, idx_val)
        loss_val_list.append(loss_val.item())

        if epoch%10==1:
            print('Epoch: {:04d}'.format(epoch+1), 'loss_train: {:.4f}'.format(loss_train.item()), 'acc_train: {:.4f}'.format(acc_train),
                'loss_val: {:.4f}'.format(loss_val.item()),'acc_val: {:.4f}'.format(acc_val))
        if epoch%100==99:
            cur_lr = 0.5 * cur_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = cur_lr

        if epoch > 200 and loss_val_list[-1] > np.mean(loss_val_list[-(early_stopping+1):-1]):
            break 

    # Test model
    model.eval()
    output = model(features, candidate_adj)

    loss_test, acc_test = evaluate(output, labels, idx_test)
    print("Train accuracy= {:.4f}".format(acc_train), "Val accuracy= {:.4f}".format(acc_val), "Test accuracy= {:.4f}".format(acc_test), "epoch= {:04d}".format(epoch))

    return acc_test

adj, features, labels, idx_train, idx_val, idx_test, flag_direct, population_test, select_index = read_data(args.data)

# Initialize incsearch and the motif adj matrix
search_base = np.array(adj.toarray(),dtype=np.int32) # dense array of base adj
print('Dataset contains:',len(search_base),'nodes,', sum(sum(search_base)), 'edges.')

node_num = len(search_base)
search.init_incsearch_model(search_base, flag_direct, flag_acc)
adj_dic = {}
init_motif = np.zeros((2, 2), dtype=np.int32)

# adj = normalize(adj)
if flag_direct:
    init_motif[1, 0] = 1
    adj_dic[str(list(init_motif.flatten()))] = [sparse_mx_to_torch_sparse_tensor(sp.eye(node_num)), sparse_mx_to_torch_sparse_tensor(adj)] # self-loop
    init_motif[0, 1] = 1
    init_motif[1, 0] = 0
    adj_dic[str(list(init_motif.flatten()))] = [sparse_mx_to_torch_sparse_tensor(sp.eye(node_num)), sparse_mx_to_torch_sparse_tensor(adj.T)]    
else:
    init_motif[0, 1] = 1
    init_motif[1, 0] = 1
    adj_dic[str(list(init_motif.flatten()))] = [sparse_mx_to_torch_sparse_tensor(sp.eye(node_num)), sparse_mx_to_torch_sparse_tensor(adj)]

motifadj_test, adj_dic = construct_motif_adj_batch([population_test], adj_dic, search_base, flag_direct, flag_acc)
motifadj_test = motifadj_test[0]
motifadj_test = [motifadj_test[ind] for ind in select_index]

test_score = []
for ind in range(test_run):
    id_list = range(node_num)
    random.shuffle(id_list)
    id_len = len(id_list)
    idx_train = id_list[:int(id_len*0.6)]
    idx_val = id_list[int(id_len*0.6):int(id_len*0.8)]
    idx_test = id_list[int(id_len*0.8):]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    acc = train_model(nlayer, nepoch, motifadj_test, features, labels, idx_train, idx_val, idx_test, attn, float(args.lr), float(args.wd), float(args.dropout), int(args.hid))
    test_score.append(acc)

test_acc_mean, test_acc_std = np.mean(test_score), np.std(test_score)
print('Final result:', test_acc_mean, test_acc_std)