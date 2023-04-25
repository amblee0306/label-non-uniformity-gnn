import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import random
from dgl.data import CoauthorCSDataset, AmazonCoBuyPhotoDataset, AmazonCoBuyComputerDataset, CoauthorPhysicsDataset
from torch_geometric.datasets import WebKB, WikipediaNetwork
import pickle


def get_data(name, split=0):  
    path = 'data/splits'  
    if name in ['cornell','wisconsin','texas']:   
        dataset = WebKB(path,name=name)    
        splits_file = np.load(f'{path}/{name}_split_0.6_0.2_{split}.npz')  
    elif name in ["chameleon", "crocodile", "squirrel"]:   
        dataset = WikipediaNetwork(path,name=name)    
        splits_file = np.load(f'{path}/{name}_split_0.6_0.2_{split}.npz')    
    data = dataset[0]   
    print('data', data)
    train_mask = splits_file['train_mask']  
    val_mask = splits_file['val_mask']  
    test_mask = splits_file['test_mask']  
    data.train_mask = torch.tensor(train_mask, dtype=torch.bool)  
    data.val_mask = torch.tensor(val_mask, dtype=torch.bool)  
    data.test_mask = torch.tensor(test_mask, dtype=torch.bool)  
    return data


def save_graph(args, best_eta_1, best_eta_2, best_G_prime_edge_tuple):
    filename = args.dataset + '-model-'+ str(args.model) + '-eta1-' + str(best_eta_1) + '-eta2-' + str(best_eta_2) + '.pkl'
    file = open(filename, 'wb')
    pickle.dump(best_G_prime_edge_tuple, file)
    file.close()
    file2 = open(filename, 'rb')
    G_prime_edge_tuple_load = pickle.load(file2)
    assert G_prime_edge_tuple_load == best_G_prime_edge_tuple
    print('saved successfully as ', filename)


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if int(args.gpu) >= 0:
        torch.cuda.manual_seed(args.seed)

def set_masks(args):
    args.test_mask = None
    args.train_mask = None
    args.val_mask = None
    if args.dataset in ['photo', 'computer', 'cs', 'physics']:
        if args.dataset == 'photo':    
            data = AmazonCoBuyPhotoDataset()
        elif args.dataset == 'computer':
            data = AmazonCoBuyComputerDataset()
        elif args.dataset == 'physics':
            data = CoauthorPhysicsDataset()
        elif args.dataset == 'cs':
            data = CoauthorCSDataset()
        g = data[0]
        _seed = 1234
        random_state = np.random.RandomState(_seed)
        features = g.ndata['feat']
        labels = g.ndata['label']
        in_feats = features.shape[1]
        n_classes = data.num_classes
        split = {'train_examples_per_class':20, 
                    'val_size':500,
                    'test_size': 1000}
        one_hot_labels = F.one_hot(labels, num_classes=n_classes).cpu().numpy()
        train_indices, val_indices, test_indices = get_train_val_test_split(random_state, one_hot_labels, **split)
        
        train_mask = torch.full((g.number_of_nodes(),), False)
        train_mask[train_indices] = True
        val_mask = torch.full((g.number_of_nodes(),), False)
        val_mask[val_indices] = True
        test_mask = torch.full((g.number_of_nodes(),), False)
        test_mask[test_indices] = True

        args.test_mask = test_mask
        args.train_mask = train_mask
        args.val_mask = val_mask


def normalisation(g, cuda):
  degs = g.in_degrees().float()
  norm = torch.pow(degs, -0.5)
  norm[torch.isinf(norm)] = 0
  if cuda:
      norm = norm.cuda()
  g.ndata['norm'] = norm.unsqueeze(1)
  return g


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits_full = model(features)
        logits = logits_full[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels), logits_full


def generate_new_train_set(logits_test, test_mask, train_mask, groundtruth_labels, ETA_1 = 10):
    import networkx as nx
    # CONSTANTS
    ETA_1 = ETA_1/100 # in percentages
    n_class = logits_test.size()[1]
    
    # STEP (a) # softmax the logits
    m = nn.Softmax(dim=1)
    mu_v = m(logits_test) 
    c = 1/n_class
    w_v = torch.sum(torch.abs(mu_v - c), dim=1) # broadcast, abs, sum

    #####################################
    # THIS IS THE LEARNED LABELS PREVIOUSLY
    learned_labels_full = m(logits_test) 
    y_learned_full = torch.argmax(learned_labels_full, dim=1)
    #####################################

    # STEP (b)
    _, indices = torch.sort(w_v, descending=True) # indices for masking purpose later # BIG TO SMALL

    ###########################################
    original_test_indices_list = list((test_mask == True).nonzero(as_tuple=True)[0])
    original_test_indices_set = set([int(idx) for idx in original_test_indices_list])
    indices_list = list(indices)
    sorted_indices = [int(idx) for idx in indices_list]
    sorted_indices_new = []
    for element in sorted_indices:
        # ONLY ADD NODES FROM THE TEST SET TO THE NEW TRAIN SET. 
        if element in original_test_indices_set:
            sorted_indices_new.append(element)
    ###########################################

    # STEP (c) # GET THE x% of LARGEST VALUES' NODES
    # last few items in the list
    eta_fraction_nodes = int(len(sorted_indices_new) * ETA_1)
    V_prime_indices = sorted_indices_new[:eta_fraction_nodes] 
    # STEP (d)
    groundtruth_labels[V_prime_indices] = y_learned_full[V_prime_indices]
    train_mask[V_prime_indices] = True
    return groundtruth_labels, train_mask


def generate_g_prime(logits_test, g_original, ETA_1 = 50, ETA_2 = 50):
  import dgl
  import networkx as nx
  # CONSTANTS
  ETA_1 = ETA_1/100 # in percentages
  ETA_2 = ETA_2/100 
  n_class = logits_test.size()[1]

  # STEP (a)
  # softmax the logits
  m = nn.Softmax(dim=1)
  mu_v = m(logits_test)
  c = 1/n_class
  w_v = torch.sum(torch.abs(mu_v - c), dim=1) # broadcast, abs, sum

  # STEP (b)
  sorted_w_v, indices = torch.sort(w_v) # indices for masking purpose later

  # STEP (c)
  eta_fraction_nodes = int(indices.size()[0] * ETA_1)
  V_prime = indices[:eta_fraction_nodes]

  # STEP (d)
  # form induced subgraph G_V_prime
  nxg_original = dgl.to_networkx(g_original.cpu()).to_undirected() # node indexing should be the same as dgl
  nxg_G_V_prime = nx.induced_subgraph(nxg_original, V_prime.tolist()).copy()

  # form spanning tree T_V_prime (but there are multiple connected components)
  # one spanning tree for each connected component

  # determine each component first
  S = [nxg_G_V_prime.subgraph(c).copy() for c in nx.connected_components(nxg_G_V_prime)]

  # apply on each component
  T_V_prime_set = set()
  E_V_prime_c = set()
  for nxg_component in S:
    node_list = list(nxg_component.nodes)
    GV_prime_component_based_edges = [frozenset(edge) for edge in list(nxg_component.edges())]
    if len(node_list) > 1: # only when the component has more than one node
      start_i = list(nxg_component.nodes)[0]
      bfs_tree_generated = nx.bfs_tree(nxg_component, start_i)
      assert len(list(bfs_tree_generated.edges())) <= len(list(nxg_component.edges)) # edges of TV' vs edges GV' (for that component)
      
      TV_prime_edges = [frozenset(edge) for edge in list(bfs_tree_generated.edges())]
      T_V_prime_set.update(TV_prime_edges)

      # outside of TV' but in GV'
      E_V_prime = set(GV_prime_component_based_edges).difference(set(TV_prime_edges))
      E_V_prime_c.update(E_V_prime)

  # STEP E
  from random import sample
  number_of_edges_to_keep = len(E_V_prime_c) - int(len(E_V_prime_c)*ETA_2)
  edges_to_keep = sample(E_V_prime_c, number_of_edges_to_keep)

  # STEP F(a) # ambient_edges
  # STEP F(b) # T_V_prime_set
  # STEP F(c) # edges_to_keep
  original_edges = [frozenset(edge) for edge in list(nxg_original.edges())]
  GV_prime_edges = [frozenset(edge) for edge in list(nxg_G_V_prime.edges())]
  ambient_edges = set(original_edges).difference(GV_prime_edges)

  G_prime_edge_set = set()
  G_prime_edge_set.update(ambient_edges)
  G_prime_edge_set.update(T_V_prime_set)
  G_prime_edge_set.update(edges_to_keep)

  # print('original edges', len(original_edges), 'new edges', len(G_prime_edge_set))

  # Convert back to dgl graph to feed into model
  G_prime_edge_tuple = [tuple(list(edge)) for edge in G_prime_edge_set]
  nxg_G_prime=nx.Graph()
  nxg_G_prime.add_nodes_from(nxg_original)
  nxg_G_prime.add_edges_from(G_prime_edge_tuple) 
  g_new = dgl.from_networkx(nxg_G_prime)
  print('graph generated: ', g_new)
  return g_new, G_prime_edge_tuple


def get_train_val_test_split(random_state,
                             labels,
                             train_examples_per_class=None, val_examples_per_class=None,
                             test_examples_per_class=None,
                             train_size=None, val_size=None, test_size=None):
    num_samples, num_classes = labels.shape
    remaining_indices = list(range(num_samples))

    if train_examples_per_class is not None:
        train_indices = sample_per_class(random_state, labels, train_examples_per_class)
    else:
        # select train examples with no respect to class distribution
        train_indices = random_state.choice(remaining_indices, train_size, replace=False)

    if val_examples_per_class is not None:
        val_indices = sample_per_class(random_state, labels, val_examples_per_class, forbidden_indices=train_indices)
    else:
        remaining_indices = np.setdiff1d(remaining_indices, train_indices)
        val_indices = random_state.choice(remaining_indices, val_size, replace=False)

    forbidden_indices = np.concatenate((train_indices, val_indices))
    if test_examples_per_class is not None:
        test_indices = sample_per_class(random_state, labels, test_examples_per_class,
                                        forbidden_indices=forbidden_indices)
    elif test_size is not None:
        remaining_indices = np.setdiff1d(remaining_indices, forbidden_indices)
        test_indices = random_state.choice(remaining_indices, test_size, replace=False)
    else:
        test_indices = np.setdiff1d(remaining_indices, forbidden_indices)

    # assert that there are no duplicates in sets
    assert len(set(train_indices)) == len(train_indices)
    assert len(set(val_indices)) == len(val_indices)
    assert len(set(test_indices)) == len(test_indices)
    # assert sets are mutually exclusive
    assert len(set(train_indices) - set(val_indices)) == len(set(train_indices))
    assert len(set(train_indices) - set(test_indices)) == len(set(train_indices))
    assert len(set(val_indices) - set(test_indices)) == len(set(val_indices))
    if test_size is None and test_examples_per_class is None:
        # all indices must be part of the split
        assert len(np.concatenate((train_indices, val_indices, test_indices))) == num_samples

    if train_examples_per_class is not None:
        train_labels = labels[train_indices, :]
        train_sum = np.sum(train_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(train_sum).size == 1

    if val_examples_per_class is not None:
        val_labels = labels[val_indices, :]
        val_sum = np.sum(val_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(val_sum).size == 1

    if test_examples_per_class is not None:
        test_labels = labels[test_indices, :]
        test_sum = np.sum(test_labels, axis=0)
        # assert all classes have equal cardinality
        assert np.unique(test_sum).size == 1

    return train_indices, val_indices, test_indices


def sample_per_class(random_state, labels, num_examples_per_class, forbidden_indices=None):
    num_samples, num_classes = labels.shape
    sample_indices_per_class = {index: [] for index in range(num_classes)}

    # get indices sorted by class
    for class_index in range(num_classes):
        for sample_index in range(num_samples):
            if labels[sample_index, class_index] > 0.0:
                if forbidden_indices is None or sample_index not in forbidden_indices:
                    sample_indices_per_class[class_index].append(sample_index)

    # get specified number of indices for each class
    return np.concatenate(
        [random_state.choice(sample_indices_per_class[class_index], num_examples_per_class, replace=False)
         for class_index in range(len(sample_indices_per_class))
         ])

