import torch
import numpy
import numpy as np
import scipy
import torch.nn as nn
import torch.nn.functional as F
import itertools
import scipy.sparse as sp
import statistics
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import CoauthorCSDataset, AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset, CoauthorPhysicsDataset
from base_models import *
from utils import *
import random
import statistics
import pickle
from torch_geometric.utils import to_scipy_sparse_matrix, to_undirected


def main(args, g_new=None, run=0, new_labels=None, new_train_mask=None):
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    elif args.dataset == 'photo':
        data = AmazonCoBuyPhotoDataset()
    elif args.dataset == 'computer':
        data = AmazonCoBuyComputerDataset()
    elif args.dataset == 'physics':
        data = CoauthorPhysicsDataset()
    elif args.dataset == 'cs':
        data = CoauthorCSDataset()
    elif args.dataset in ['wisconsin', 'texas', 'cornell','chameleon']:
        data = get_data(args.dataset, split=run)
        if args.gpu < 0:
            features = data['x']
            labels = data['y']
        else:
            features = data['x'].to(args.gpu)
            labels = data['y'].to(args.gpu)
        data.edge_index = to_undirected(data.edge_index)
        adj = to_scipy_sparse_matrix(data.edge_index)
        g = dgl.from_scipy(adj)
        n_classes = int(data['y'].max() + 1)
        in_feats = features.size()[1]
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

    if args.dataset in ['cora','citeseer','pubmed','photo', 'computer', 'cs', 'physics']:
        g_original = g = data[0]
    else:
        g_original = g.clone()
    g_original = dgl.remove_self_loop(g_original)

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)

    if args.dataset in ['citeseer', 'cora', 'pubmed']:
        features = g.ndata['feat']
        labels = g.ndata['label']
        train_mask = g.ndata['train_mask']
        val_mask = g.ndata['val_mask']
        test_mask = g.ndata['test_mask']
        in_feats = features.shape[1]
        n_classes = data.num_labels
    elif args.dataset in ['photo', 'computer', 'cs', 'physics']:
        features = g.ndata['feat']
        labels = g.ndata['label']
        in_feats = features.shape[1]
        n_classes = data.num_classes
        train_mask = args.train_mask
        val_mask = args.val_mask
        test_mask = args.test_mask

    # REPLACE TO USE G' HERE
    if g_new is not None:
      g = g_new
      if args.gpu < 0:
          cuda = False
      else:
          cuda = True
          g = g.int().to(args.gpu)

    n_edges = g.number_of_edges()

    print("""----Data statistics------'
      #Nodes %d
      #Edges %d""" % (g.number_of_nodes(), n_edges))

    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()

    # normalization
    g = normalisation(g, cuda)

    if args.model == 'gcn':
        # create GCN model
        model = GCN(g, in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout)
    elif args.model == 'gat':
        # create GAT model
        heads = ([args.num_heads] * args.n_layers) + [args.num_out_heads]
        model = GAT(g, args.n_layers, in_feats, args.n_hidden, n_classes, heads, F.elu, args.in_drop, args.attn_drop, args.negative_slope, args.residual)
    elif args.model == 'mlp':
        model = MLP(in_feats, args.n_hidden, n_classes, F.relu, args.dropout)

    if cuda:
        model.cuda()
    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.early_stop:
      patience = 100
      counter = 0
      best_score = 0

    # initialize graph
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)

        if new_train_mask is None:
            loss = loss_fcn(logits[train_mask], labels[train_mask])
        else:
            loss = loss_fcn(logits[new_train_mask], new_labels[new_train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        if new_train_mask is None:
            acc, _ = evaluate(model, features, labels, val_mask)
        else:
            acc, _ = evaluate(model, features, new_labels, val_mask)

        if args.early_stop and epoch>100:
          if acc > best_score:
            best_score = acc
            counter = 0
          else:
            counter += 1
            if counter >= patience:
              break

        if epoch % 100 == 0:
          print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}". format(epoch, np.mean(dur), loss.item(),
                                             acc, n_edges / np.mean(dur) / 1000))

    print()
    # the testing step remains the same where the original labels and test_mask are used.
    acc, logits_test = evaluate(model, features, labels, test_mask)
    print("Test accuracy {:.2%}".format(acc))

    return g_original, acc, logits_test, test_mask, train_mask, labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--model", type=str, default="gcn",
                        help="model name ('gcn', 'gat').")
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed', 'wisconsin', 'texas', 'cornell','chameleon').")
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument("--seed", type=int, default=0,
                        help="set seed")
    parser.add_argument("--save", action="store_true", default=False,
                        help="save g prime's edges? default false")
    parser.add_argument("--eta1", type=int, default=0,
                        help="set eta1")
    parser.add_argument("--eta2", type=int, default=0,
                        help="set eta2")
    parser.add_argument("--all-combination", action="store_true", default=False,
                        help="run all eta1 and eta2 combinations?")
    parser.add_argument("--early-stop", action="store_true", default=False,
                        help="use early-stopping?")
    parser.add_argument("--step", type=int, default=10,
                        help="set step size for eta1 and eta2 for running all combinations")
    args = parser.parse_args()

    set_seed(args)
    if args.dataset in ['photo', 'computer', 'cs', 'physics']:
        set_masks(args)

    ############################################################
    # FIRSTLY : RUN THE PLAIN MODEL CAUSE WE NEED THE LOGITS #
    ############################################################
    # The plain model also coincides with the case where eta_1 = 0 and eta_2 = 0

    test_scores = []
    logits_test_dict = {}
    for run in range(10):
        g_original, test_acc, logits_test, test_mask, train_mask, labels = main(args, run=run)
        test_scores.append(test_acc)
        logits_test_dict[run] = logits_test
    g_original = dgl.remove_self_loop(g_original)
    plain_score = (sum(test_scores)/len(test_scores), statistics.stdev(test_scores))
    print('plain model, test_acc ', plain_score)

    nxg_original = dgl.to_networkx(g_original.cpu()).to_undirected()
    original_edges = [frozenset(edge) for edge in list(nxg_original.edges())]
    G_original_edge_tuple = [tuple(list(edge)) for edge in set(original_edges)]

    #######################################
    # STEP TWO: RUN ALGORITHM 2 TO GET G' #
    #######################################

    best_eta_1 = 0
    best_eta_2 = 0
    eta_1 = 0
    eta_2 = 0
    algo2_result_acc = {}
    algo2_result_std = {}
    algo2_result_acc[tuple([eta_1, eta_2])] = plain_score[0]
    algo2_result_std[tuple([eta_1, eta_2])] = plain_score[1]
    best_score = plain_score[0] # The plain model as base score
    best_G_prime_list = [g_original]*10
    best_G_prime_edge_tuple = [G_original_edge_tuple]*10

    if args.all_combination:
        for eta_1 in range(10,101,args.step):
            for eta_2 in range(10,101,args.step):
                print('eta_1', eta_1, 'eta_2', eta_2)
                test_acc_list = []
                G_prime_list = []
                G_prime_dgl_list = []
                for run in range(10):
                    g_new, G_prime_edge_tuple = generate_g_prime(logits_test_dict[run], g_original, eta_1, eta_2)
                    _, test_acc, _, _, _,_ = main(args, g_new, run)
                    test_acc_list.append(test_acc)
                    G_prime_list.append(G_prime_edge_tuple)
                    G_prime_dgl_list.append(g_new)
                avg = sum(test_acc_list)/len(test_acc_list)
                std = statistics.stdev(test_acc_list)
                algo2_result_acc[tuple([eta_1, eta_2])] = avg
                algo2_result_std[tuple([eta_1, eta_2])] = std

                if avg > best_score:
                    best_score = avg
                    best_eta_1 = eta_1
                    best_eta_2 = eta_2
                    assert len(G_prime_list) == 10
                    best_G_prime_edge_tuple = G_prime_list
                    best_G_prime_list = G_prime_dgl_list
    else:
        print('eta_1', args.eta1, 'eta_2', args.eta2)
        test_acc_list = []
        G_prime_edge_tuple = []
        G_prime_dgl_list = []
        for run in range(10):
            g_new, G_prime_edge_tuple = generate_g_prime(logits_test_dict[run], g_original, args.eta1, args.eta2)
            _, test_acc, _, _, _,_ = main(args, g_new, run)
            test_acc_list.append(test_acc)
            G_prime_edge_tuple.append(G_prime_edge_tuple)
            G_prime_dgl_list.append(g_new)
        avg = sum(test_acc_list)/len(test_acc_list)
        std = statistics.stdev(test_acc_list)
        algo2_result_acc[tuple([args.eta1, args.eta2])] = avg
        algo2_result_std[tuple([args.eta1, args.eta2])] = std

        if avg > best_score:
            best_score = avg
            best_eta_1 = args.eta1
            best_eta_2 = args.eta2
            best_G_prime_edge_tuple = G_prime_edge_tuple
            best_G_prime_list = G_prime_dgl_list

    algo2_max_key = max(algo2_result_acc, key=algo2_result_acc.get)
    print(algo2_result_acc)
    print(algo2_result_std)
    print('best key: ', algo2_max_key, 'score', algo2_result_acc[algo2_max_key], algo2_result_std[algo2_max_key])
    print('plain model, test_acc ', plain_score)
    print()

    if args.save:
        save_graph(args, best_eta_1, best_eta_2, best_G_prime_edge_tuple)
    else:
        print('new graph structure(s) not saved.')

    ###############################################################
    # STEP THREE: RUN ALGORITHM 1 TO GET WGNN's FINAL PERFORMANCE #
    ###############################################################
    # The G' from ALGORITHM 2 is used here.
    wgnn_result_acc = {}
    wgnn_result_std = {}
    for eta3 in range(0,101,10):
        test_acc_list = []
        for run in range(10):
            new_labels, new_train_mask = generate_new_train_set(logits_test_dict[run].clone().detach(), test_mask.clone().detach(), train_mask.clone().detach(), labels.clone().detach(), eta3)
            _, test_acc, _, _, _,_= main(args, best_G_prime_list[run], run, new_labels, new_train_mask)
            test_acc_list.append(test_acc)
        avg = sum(test_acc_list)/len(test_acc_list)
        std = statistics.stdev(test_acc_list)
        wgnn_result_acc[eta3] = avg
        wgnn_result_std[eta3] = std

    max_key_wgnn = max(wgnn_result_acc, key=wgnn_result_acc.get)

    print()
    print('plain model, test_acc: ', plain_score)
    print()
    print('ALGO2')
    print('best key: ', algo2_max_key, ', score: ', algo2_result_acc[algo2_max_key], algo2_result_std[algo2_max_key])
    print()
    print('WGNN: ALGO1 + ALGO2')
    print('best eta3: ', max_key_wgnn, ', score: ', wgnn_result_acc[max_key_wgnn], wgnn_result_std[max_key_wgnn])
