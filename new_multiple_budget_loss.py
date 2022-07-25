import numpy as np
import argparse
import os
import collections
import json
import queue
import time
import LogLossCalculator
import MultipleBudgetEdgeMatrix

from utils.data_utils import load_dataset_numpy

import scipy.spatial.distance

from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import maximum_flow
from utils.flow import _make_edge_pointers

from cvxopt import solvers, matrix, spdiag, log, mul, sparse, spmatrix

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

args = parser.parse_args("--dataset_in=MNIST --num_samples=2500 --use_full --eps1 0 0 0 0 0 0 --eps2 0 1 2 3 4 5".split())

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
    subsample_sizes = [500, 800, 2500,3200, args.num_samples]


rng = np.random.default_rng(100)
eps= []
for i in range(len(args.eps1)):
    eps.append([float(args.eps1[i]), float(args.eps2[i])])

loss_final=[]
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
    final_loss_matrix_store=[]
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
                edge_matrix, total_edge_matrix, total_edges = Multiple_Budget_Edge_Matrix.build_edge_matrix(D_12, eps)
                total_edge_matrix_budget[tuple(eps)] = total_edge_matrix
                total_edges_budget.append((eps, total_edges))
                final_edge_matrix[tuple(eps)] = edge_matrix
                if args.run_generic:
                     n_1, n_2,time_taken,output = LogLossCalculator.optimal_log_loss_cvxopt(edge_matrix,
                                                                                                      subsample_size,
                                                                                                      subsample_size,len(eps))
                     time_generic_list.append(time_taken)
                else:
                 final_classifier_probs, n_1, n_2,time_taken = LogLossCalculator.optimal_log_loss(edge_matrix,subsample_size , subsample_size,len(eps))
                 final_loss, final_loss_matrix= LogLossCalculator.log_loss(final_classifier_probs, n_1, n_2, 2)
                 loss_final.append([eps, final_loss, subsample_size])
                 loss_list.append(final_loss)
                 final_loss_matrix_store.append([eps,final_loss_matrix])
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
