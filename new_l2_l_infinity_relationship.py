import numpy as np
import argparse
import time
import os
import collections
import json
import queue
import time
import LogLossCalculator
import L2L_infinityEdgeMatrix

from utils.data_utils import load_dataset_numpy

import scipy.spatial.distance

from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import maximum_flow
from utils.flow import _make_edge_pointers

from cvxopt import solvers, matrix, spdiag, log, mul, sparse, spmatrix


def minll(G, h, p):
    m, v_in = G.size

    def F(x=None, z=None):
        if x is None:
            return 0, matrix(1.0, (v, 1))
        if min(x) <= 0.0:
            return None
        f = -sum(mul(p, log(x)))
        Df = mul(p.T, -(x ** -1).T)
        if z is None:
            return f, Df
        # Fix the Hessian
        H = spdiag(z[0] * mul(p, x ** -2))
        return f, Df, H

    return solvers.cp(F, G=G, h=h)


def find_remaining_cap_edges(edge_ptr, capacities, heads, tails, source, sink):
    ITYPE = np.int32
    n_verts = edge_ptr.shape[0] - 1
    n_edges = capacities.shape[0]
    ITYPE_MAX = np.iinfo(ITYPE).max

    # Our result array will keep track of the flow along each edge
    flow = np.zeros(n_edges, dtype=ITYPE)

    # Create a circular queue for breadth-first search. Elements are
    # popped dequeued at index start and queued at index end.
    q = np.empty(n_verts, dtype=ITYPE)

    # Create an array indexing predecessor edges
    pred_edge = np.empty(n_verts, dtype=ITYPE)

    # While augmenting paths from source to sink exist
    for k in range(n_verts):
        pred_edge[k] = -1
    path_edges = []
    # Reset queue to consist only of source
    q[0] = source
    start = 0
    end = 1
    # While we have not found a path, and queue is not empty
    path_found = False
    while start != end and not path_found:
        # Pop queue
        cur = q[start]
        start += 1
        if start == n_verts:
            start = 0
        # Loop over all edges from the current vertex
        for e in range(edge_ptr[cur], edge_ptr[cur + 1]):
            t = heads[e]
            if pred_edge[t] == -1 and t != source and \
                    capacities[e] > flow[e]:
                pred_edge[t] = e
                path_edges.append((cur, t))
                if t == sink:
                    path_found = True
                    break
                # Push to queue
                q[end] = t
                end += 1
                if end == n_verts:
                    end = 0
    return path_edges


def create_graph_rep(edge_matrix, n_1, n_2):
    graph_rep = []
    for i in range(n_1 + n_2 + 2):
        graph_rep.append([])
        if i == 0:
            # source
            for j in range(n_1 + n_2 + 2):
                if j == 0:
                    graph_rep[i].append(0)
                elif 1 <= j <= n_1:
                    graph_rep[i].append(n_2)
                elif n_1 < j <= n_1 + n_2 + 1:
                    graph_rep[i].append(0)
        elif 1 <= i <= n_1:
            # LHS vertices
            for j in range(n_1 + n_2 + 2):
                if j <= n_1:
                    graph_rep[i].append(0)
                elif n_1 < j <= n_1 + n_2:
                    if edge_matrix[i - 1, j - n_1 - 1]:
                        graph_rep[i].append(n_1 * n_2)
                    else:
                        graph_rep[i].append(0)
                elif n_1 + n_2 < j:
                    graph_rep[i].append(0)
        elif n_1 < i <= n_1 + n_2:
            # RHS vertices
            for j in range(n_1 + n_2 + 2):
                if j <= n_1 + n_2:
                    graph_rep[i].append(0)
                elif j > n_1 + n_2:
                    graph_rep[i].append(n_1)
        elif i == n_1 + n_2 + 1:
            # Sink
            for j in range(n_1 + n_2 + 2):
                graph_rep[i].append(0)

    graph_rep_array = np.array(graph_rep)

    return graph_rep_array


def set_classifier_prob_full_flow(top_level_vertices, n_1_curr, n_2_curr, sink_idx):
    for item in top_level_vertices:
        if item != 0 and item != sink_idx:
            classifier_probs[item - 1, 0] = n_1_curr / (n_1_curr + n_2_curr)
            classifier_probs[item - 1, 1] = n_2_curr / (n_1_curr + n_2_curr)


def set_classifier_prob_no_flow(top_level_vertices, n_1, sink_idx):
    for item in top_level_vertices:
        if item != 0 and item != sink_idx:
            if item <= n_1:
                classifier_probs[item - 1, 0] = 1
                classifier_probs[item - 1, 1] = 0
            elif item > n_1:
                classifier_probs[item - 1, 0] = 0
                classifier_probs[item - 1, 1] = 1


def graph_rescale(graph_rep_curr, top_level_indices, n_1, n_2):
    n_1_curr = len(np.where(top_level_indices <= n_1)[0]) - 1
    n_2_curr = len(np.where(top_level_indices > n_1)[0]) - 1
    # source rescale
    # print(graph_rep_curr[0])
    graph_rep_curr[0, :] = graph_rep_curr[0, :] / n_2
    graph_rep_curr[0, :] *= n_2_curr
    # print(graph_rep_curr[0])
    # bipartite graph edge scale
    graph_rep_curr[1:n_1_curr + 1, :] = graph_rep_curr[1:n_1_curr + 1, :] / (n_1 * n_2)
    graph_rep_curr[1:n_1_curr + 1, :] *= (n_1_curr * n_2_curr)
    # sink edges rescale
    graph_rep_curr[n_1_curr + 1:, :] = graph_rep_curr[n_1_curr + 1:, :] / n_1
    graph_rep_curr[n_1_curr + 1:, :] *= n_1_curr
    return graph_rep_curr, n_1_curr, n_2_curr


def find_flow_and_split(top_level_indices, graph_rep_array, n_1, n_2, sink_idx):
    top_level_indices_1 = None
    top_level_indices_2 = None
    # Create subgraph from index array provided
    graph_rep_curr = graph_rep_array[top_level_indices]
    graph_rep_curr = graph_rep_curr[:, top_level_indices]
    graph_rep_curr, n_1_curr, n_2_curr = graph_rescale(graph_rep_curr, top_level_indices)
    graph_curr = csr_matrix(graph_rep_curr)
    flow_curr = maximum_flow(graph_curr, 0, len(top_level_indices) - 1)
    # Checking if full flow occurred, so no need to split
    if flow_curr.flow_value == n_1_curr * n_2_curr:
        set_classifier_prob_full_flow(top_level_indices, n_1_curr, n_2_curr)
        return top_level_indices_1, top_level_indices_2, flow_curr
    elif flow_curr.flow_value == 0:
        set_classifier_prob_no_flow(top_level_indices)
        return top_level_indices_1, top_level_indices_2, flow_curr
    # Finding remaining capacity edges
    remainder_array = graph_curr - flow_curr.residual

    rev_edge_ptr, tails = _make_edge_pointers(remainder_array)

    edge_ptr = remainder_array.indptr
    capacities = remainder_array.data
    heads = remainder_array.indices

    edge_list_curr = find_remaining_cap_edges(edge_ptr, capacities, heads, tails, 0, len(top_level_indices) - 1)

    #     print(edge_list_curr)
    gz_idx = []
    for item in edge_list_curr:
        gz_idx.append(item[0])
        gz_idx.append(item[1])
    if len(gz_idx) > 0:
        gz_idx = np.array(gz_idx)
        gz_idx_unique = np.unique(gz_idx)
        top_level_gz_idx = top_level_indices[gz_idx_unique]
        top_level_gz_idx = np.insert(top_level_gz_idx, len(top_level_gz_idx), sink_idx)
        top_level_indices_1 = top_level_gz_idx
    else:
        top_level_gz_idx = np.array([0, sink_idx])
    # Indices without flow
    top_level_z_idx = np.setdiff1d(top_level_indices, top_level_gz_idx)
    if len(top_level_z_idx) > 0:
        # Add source and sink back to zero flow idx array
        top_level_z_idx = np.insert(top_level_z_idx, 0, 0)
        top_level_z_idx = np.insert(top_level_z_idx, len(top_level_z_idx), sink_idx)
        top_level_indices_2 = top_level_z_idx

    return top_level_indices_1, top_level_indices_2, flow_curr


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_in", default='MNIST',
                    help="dataset to be used")
parser.add_argument("--norm", default='l2',
                    help="norm to be used")
parser.add_argument('--num_samples', type=int, default=None)
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument("--eps1", nargs="+", default=["0"])
parser.add_argument("--eps2", nargs="+", default=["0"])
parser.add_argument('--approx_only', dest='approx_only', action='store_true')
parser.add_argument('--use_test', dest='use_test', action='store_true')
parser.add_argument('--track_hard', dest='track_hard', action='store_true')
parser.add_argument('--use_full', dest='use_full', action='store_true')
parser.add_argument('--run_generic', dest='run_generic', action='store_true')
parser.add_argument('--new_marking_strat', type=str, default=None)
parser.add_argument('--num_reps', type=int, default=2)
parser.add_argument('--class_1', type=int, default=3)
parser.add_argument('--class_2', type=int, default=7)

args = parser.parse_args(
    "--dataset_in=MNIST --num_samples=2500 --use_full --eps1 0 0 0 0 0 0 --eps2 0 1 2 3 4 5".split())

train_data, test_data, data_details = load_dataset_numpy(args, data_dir='data',
                                                         training_time=False)
DATA_DIM = data_details['n_channels'] * data_details['h_in'] * data_details['w_in']

X = []
Y = []

# Pytorch normalizes tensors (so need manual here!)
if args.use_test:
    for (x, y, _, _, _) in test_data:
        X.append(x / 255.)
        Y.append(y)
else:
    for (x, y, _, _, _) in train_data:
        X.append(x / 255.)
        Y.append(y)

X = np.array(X)
Y = np.array(Y)

num_samples = int(len(X) / 2)
print(num_samples)

X_c1 = X[:num_samples].reshape(num_samples, DATA_DIM)
X_c2 = X[num_samples:].reshape(num_samples, DATA_DIM)

class_1 = args.class_1
class_2 = args.class_2

if not os.path.exists('distances'):
    os.makedirs('distances')

if not os.path.exists('cost_results'):
    os.makedirs('cost_results')

if args.use_full:
    subsample_sizes = [args.num_samples]
else:
    subsample_sizes = [500, 800, 2500, 3200, args.num_samples]

rng = np.random.default_rng(100)
eps = []
for i in range(len(args.eps1)):
    eps.append([float(args.eps1[i]), float(args.eps2[i])])

loss_final = []
for subsample_size in subsample_sizes:
    if args.use_test:
        save_file_name = 'logloss_' + str(class_1) + '_' + str(class_2) + '_' + str(
            subsample_size) + '_' + args.dataset_in + '_test_' + args.norm
    else:
        save_file_name = 'logloss_' + str(class_1) + '_' + str(class_2) + '_' + str(
            subsample_size) + '_' + args.dataset_in + '_' + args.norm

    f = open('cost_results/' + save_file_name + '.txt', 'a')
    f_time = open('cost_results/timing_results/' + save_file_name + '.txt', 'a')

    loss_list = []
    time_list = []
    num_edges_list = []
    final_loss_matrix_store = []
    if args.run_generic:
        time_generic_list = []

    if subsample_size == args.num_samples:
        num_reps = 1
    else:
        num_reps = args.num_reps

    for rep in range(num_reps):
        indices_1 = rng.integers(num_samples, size=subsample_size)
        indices_2 = rng.integers(num_samples, size=subsample_size)

        if args.use_full:
            X_c1_curr = X_c1
            X_c2_curr = X_c2
        else:
            X_c1_curr = X_c1[indices_1]
            X_c2_curr = X_c2[indices_2]

        if args.use_test:
            dist_mat_name = args.dataset_in + '_test_' + str(class_1) + '_' + str(class_2) + '_' + str(
                subsample_size) + '_' + args.norm + '_rep' + str(rep) + '.npy'
        else:
            dist_mat_name = args.dataset_in + '_' + str(class_1) + '_' + str(class_2) + '_' + str(
                subsample_size) + '_' + args.norm + '_rep' + str(rep) + '.npy'

        if os.path.exists(dist_mat_name):
            print('Loading distances')
            D_12 = np.load('distances/' + dist_mat_name)
        else:
            if args.norm == 'l2':
                D_12 = scipy.spatial.distance.cdist(X_c1_curr, X_c2_curr, metric='euclidean')
            elif args.norm == 'l1':
                D_12 = scipy.spatial.distance.cdist(X_c1_curr, X_c2_curr, metric='cityblock')
            elif args.norm == 'linf':
                D_12 = scipy.spatial.distance.cdist(X_c1_curr, X_c2_curr, metric='chebyshev')
            np.save('distances/' + dist_mat_name, D_12)

        total_edge_matrix_budget = {}
        total_edges_budget = []
        final_edge_matrix = {}
        for eps in eps_user:
            edge_matrix, total_edge_matrix, total_edges = L2L_infinityEdgeMatrix.edge_matrix_calculator(eps,X_c1_curr,X_c2_curr)
            total_edge_matrix_budget[tuple(eps)] = total_edge_matrix
            total_edges_budget.append((eps, total_edges))
            final_edge_matrix[tuple(eps)] = edge_matrix
            if args.run_generic:
                n_1, n_2, time_taken, output = LogLossCalculator.optimal_log_loss_cvxopt(edge_matrix,
                                                                                         subsample_size,
                                                                                         subsample_size, len(eps))
                time_generic_list.append(time_taken)
            else:
                final_classifier_probs, n_1, n_2, time_taken = LogLossCalculator.optimal_log_loss(edge_matrix,
                                                                                                  subsample_size,
                                                                                                  subsample_size,
                                                                                                  len(eps))
                final_loss, final_loss_matrix = LogLossCalculator.log_loss(final_classifier_probs, n_1, n_2, 2)
                loss_final.append([eps, final_loss, subsample_size])
                loss_list.append(final_loss)
                final_loss_matrix_store.append([eps, final_loss_matrix])
                time_list.append(time_taken)
                num_edges_list.append(total_edges)

    loss_avg = np.mean(loss_list)
    loss_var = np.var(loss_list)
    time_avg = np.mean(time_list)
    time_var = np.var(time_list)
    num_edges_avg = np.mean(num_edges_list)

    f.write(str(eps) + ',' + str(loss_avg) + ',' + str(loss_var) + '\n')
    if args.run_generic:
        time_avg_generic = np.mean(time_generic_list)
        time_var_generic = np.var(time_generic_list)
        f_time.write(str(eps) + ',' + str(time_avg) + ',' + str(time_var) + ',' + str(time_avg_generic) + ',' + str(
            time_var_generic) + ',' + str(num_edges_avg) + '\n')
    else:
        f_time.write(str(eps) + ',' + str(time_avg) + ',' + str(time_var) + ',' + str(num_edges_avg) + '\n')
    np.savetxt('graph_data/optimal_probs/' + save_file_name + '_' + str(eps) + '.txt', classifier_probs, fmt='%.5f')
