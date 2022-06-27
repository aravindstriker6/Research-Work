import time
import numpy as np
import argparse
import os
import collections
import json
import queue


import scipy.spatial.distance

from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import maximum_flow
from utils.flow import _make_edge_pointers

from cvxopt import solvers, matrix, spdiag, log, mul, sparse, spmatrix

def optimal_log_loss_cvxopt(edge_matrix,class1_subsample_size,class2_subsample_size,n_blocks):
            n_1=class1_subsample_size*n_blocks
            n_2=class2_subsample_size*n_blocks
            v=(n_1+n_2)
            num_edges=len(np.where(edge_matrix==1)[0])
            edges=np.where(edge_matrix==1)
            incidence_matrix=np.zeros((num_edges,v))
            for i in range(num_edges):
                j1=edges[0][i]
                j2=edges[1][i]+(n_1-1)
                incidence_matrix[i,j1]=1
                incidence_matrix[i,j2]=1

            G_in=np.vstack((incidence_matrix,np.eye(v)))
            h_in=np.ones((num_edges+v,1))
            p=(1.0/v)*np.ones((v,1))

            G_in_sparse_np=coo_matrix(G_in)

            G_in_sparse=spmatrix(1.0,G_in_sparse_np.nonzero()[0],G_in_sparse_np.nonzero()[1])

            solvers.options['maxiters']=1000

            time_start=time()
            output=minll(G_in_sparse,matrix(h_in),matrix(p))
            print(output['primal objective'])
            time_end=time()
            if output['status'] == 'optimal':
                time_taken = (time_start-time_end)
            else:
                time_taken=(time_start-time_end)
            return n_1,n_2,time_taken
def optimal_log_loss(edge_matrix,class1_subsample_size,class2_subsample_size,n_blocks):
    n_1 = class1_subsample_size * n_blocks
    n_2 = class2_subsample_size * n_blocks
    # Create graph representation
    graph_rep_array = create_graph_rep(edge_matrix, n_1, n_2)

    time_start = time()
    q = queue.Queue()
    # Initial graph indices
    q.put(np.arange(n_1 + n_2 + 2))
    sink_idx = n_1 + n_2 + 1
    count = 0
    while not q.empty():
        #print('Current queue size at eps %s is %s' % (eps[m], q.qsize()))
        curr_idx_list = q.get()
        # print(q.qsize())
        list_1, list_2, flow_curr = find_flow_and_split(curr_idx_list)
        # print(list_1,list_2,flow_curr.flow_value)
        if list_1 is not None:
            q.put(list_1)
        if list_2 is not None:
            q.put(list_2)
    time_end = time()
    time_taken=(time_start-time_end)
    return n_1,n_2,time_taken
def log_loss(n_1,n_2):
    classifier_probs = np.zeros(((n_1) + (n_2), 2))
    loss1=0
    loss2=0
    for i in range(len(classifier_probs)):
        if i < n_1:
            loss1 += np.log(classifier_probs[i][0])
        elif i >= n_1:
            loss2 += np.log(classifier_probs[i][1])
    loss1 = -1 * loss1 / len(classifier_probs)
    loss2 = -1 * loss2 / len(classifier_probs)
    return loss1,loss2



