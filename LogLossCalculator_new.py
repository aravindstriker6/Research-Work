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
            return n_1,n_2,time_taken,output
def minll(G,h,p):
    m,v_in=G.size
    def F(x=None,z=None):
        if x is None:
            return 0, matrix(1.0,(v,1))
        if min(x)<=0.0:
            return None
        f = -sum(mul(p,log(x)))
        Df = mul(p.T,-(x**-1).T)
        if z is None:
            return f,Df
        # Fix the Hessian
        H = spdiag(z[0]*mul(p,x**-2))
        return f,Df,H
    return solvers.cp(F,G=G,h=h)
def find_remaining_cap_edges(edge_ptr,capacities,heads,tails, source, sink):
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
            if pred_edge[t] == -1 and t != source and\
                    capacities[e] > flow[e]:
                pred_edge[t] = e
                path_edges.append((cur,t))
                if t == sink:
                    path_found = True
                    break
                # Push to queue
                q[end] = t
                end += 1
                if end == n_verts:
                    end = 0
    return path_edges
def create_graph_rep(edge_matrix,n_1,n_2,weights):
    graph_rep = []
    for i in range(n_1+n_2+2):
        graph_rep.append([])
        if i==0:
            #source
            for j in range(n_1+n_2+2):
                if j==0:
                    graph_rep[i].append(0)
                elif 1<=j<=n_1:
                    graph_rep[i].append(weights[j-1])
                elif n_1<j<=n_1+n_2+1:
                    graph_rep[i].append(0)
        elif 1<=i<=n_1:
            # LHS vertices
            for j in range(n_1+n_2+2):
                if j<=n_1:
                    graph_rep[i].append(0)
                elif n_1<j<=n_1+n_2:
                    if edge_matrix[i-1,j-n_1-1]:
                        graph_rep[i].append(1)
                    else:
                        graph_rep[i].append(0)
                elif n_1+n_2<j:
                    graph_rep[i].append(0)
        elif n_1<i<=n_1+n_2:
            #RHS vertices
            for j in range(n_1+n_2+2):
                if j<=n_1+n_2:
                    graph_rep[i].append(0)
                elif j>n_1+n_2:
                    graph_rep[i].append(weights[i-1])
        elif i==n_1+n_2+1:
            #Sink
            for j in range(n_1+n_2+2):
                graph_rep[i].append(0)

    graph_rep_array=np.array(graph_rep)

    return graph_rep_array
def set_classifier_prob_full_flow(top_level_vertices,w_1_curr,w_2_curr,sink_idx):
    for item in top_level_vertices:
        if item !=0 and item != sink_idx:
            classifier_probs[item-1,0]=w_1_curr/(w_1_curr+w_2_curr)
            classifier_probs[item-1,1]=w_2_curr/(w_1_curr+w_2_curr)
def set_classifier_prob_no_flow(top_level_vertices,n_1,sink_idx):
    for item in top_level_vertices:
        if item !=0 and item != sink_idx:
            if item<=n_1:
                classifier_probs[item-1,0]=1
                classifier_probs[item-1,1]=0
            elif item>n_1:
                classifier_probs[item-1,0]=0
                classifier_probs[item-1,1]=1
def graph_rescale(graph_rep_curr,top_level_indices,n_1,n_2):
    class_1_vertices = top_level_indices[(top_level_indices > 0) & (top_level_indices <= n_1)]
    class_1_weights = weights[class_1_vertices]
    w_1_curr = np.sum(class_1_weights)
    class_2_vertices = top_level_indices[(top_level_indices > n_1) & (top_level_indices < n_1 + n_2)]
    class_2_weights = weights[class_2_vertices]
    w_2_curr = np.sum(class_2_weights)
    n_1_curr = len(class_1_vertices)
    n_2_curr = len(class_2_vertices)
    graph_rep_curr[0, :] = (graph_rep_curr[0, :] * w_2_curr)
    graph_rep_curr[1:n_1_curr + 1, :] = (graph_rep_curr[1:n_1_curr + 1, :] * (w_1_curr * w_2_curr))
    graph_rep_curr[n_1_curr + 1:, :] = (graph_rep_curr[n_1_curr + 1:, :] * w_1_curr)
    return graph_rep_curr, w_1_curr, w_2_curr


def find_flow_and_split(top_level_indices, graph_rep_array, n_1, n_2, sink_idx,weights):
    top_level_indices_1 = None
    top_level_indices_2 = None
    # Create subgraph from index array provided
    graph_rep_curr = graph_rep_array[top_level_indices]
    graph_rep_curr = graph_rep_curr[:, top_level_indices]
    graph_rep_curr,w_1_curr,w_2_curr= graph_rescale(graph_rep_curr, top_level_indices, n_1, n_2,weights)
    graph_curr = csr_matrix(graph_rep_curr)
    flow_curr = maximum_flow(graph_curr, 0, len(top_level_indices) - 1)
    # Checking if full flow occurred, so no need to split
    if flow_curr.flow_value == 0:
        set_classifier_prob_no_flow(top_level_indices, n_1, sink_idx)
        return top_level_indices_1, top_level_indices_2, flow_curr
    elif flow_curr.flow_value == w_1_curr * w_2_curr:
        set_classifier_prob_full_flow(top_level_indices, w_1_curr,w_2_curr, sink_idx)
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

def optimal_log_loss(edge_matrix,class1_subsample_size,class2_subsample_size,n_blocks):
    n_1 = class1_subsample_size * n_blocks
    n_2 = class2_subsample_size * n_blocks
    weights = np.random.randint(0, 100, n_1 + n_2)
    # Create graph representation
    graph_rep_array = create_graph_rep(edge_matrix, n_1, n_2,weights)

    time1 = time.clock()
    q = queue.Queue()
    # Initial graph indices
    q.put(np.arange(n_1 + n_2 + 2))
    sink_idx = n_1 + n_2 + 1
    global classifier_probs
    classifier_probs = np.zeros(((n_1) + (n_2), 2))
    count = 0
    while not q.empty():
        #print('Current queue size at eps %s is %s' % (eps, q.qsize()))
        curr_idx_list = q.get()
        # print(q.qsize())
        list_1, list_2, flow_curr = find_flow_and_split(curr_idx_list, graph_rep_array, n_1, n_2, sink_idx)
        # print(list_1,list_2,flow_curr.flow_value)
        if list_1 is not None:
            q.put(list_1)
        if list_2 is not None:
            q.put(list_2)
    time2 = time.clock()
    time_taken=time2-time1
    return classifier_probs, n_1, n_2,time_taken
def log_loss(final_classifier_probs,n_1,n_2,n_blocks,weights):
    loss_matrix = np.empty([n_blocks, n_blocks], dtype=float)
    target = 0
    for i in range(0, n_blocks):
        for j in range(0, int(n_1 / n_blocks)):
            if classifier_probs[j + target]==0:
                 continue
            loss_matrix[i][0] += weights[j+target]*(np.log(classifier_probs[j + target][0]))
        target = target + int(n_1 / n_blocks)
    for i in range(0, n_blocks):
        for j in range(0, int(n_2 / n_blocks)):
            if classifier_probs[j + target]==0:
                 continue
            loss_matrix[i][1] += weights[j+target]*(np.log(classifier_probs[j + target][1]))
        target = target + int(n_2 / n_blocks)
    final_loss_matrix = (-1 * loss_matrix / np.sum(weights))
    final_loss = final_loss_matrix.sum()
    return final_loss, final_loss_matrix



