import numpy as np

def build_edge_matrix(D_12,eps): # Distance matrix and epsilon list given as input to the function
 total_edge_matrix=np.empty([len(eps),len(eps)], dtype=float)
 edge_matrixes={}
 total_edges=0 # Store total number of edges
 for i in range(len(eps)):
    new=0
    for j in range(len(eps)):
                 new= D_12 <= float(eps[i])+float(eps[j]) # Check if there is an edge or not
                 new = new.astype(float)
                 total_edge_matrix[i][j]=new.sum()
                 total_edges+=new.sum()
                 edge_matrixes[(eps[i],eps[j])]=new
 stacked=[]
 for i in range(len(eps)):
        h_stack=[]
        for j in range(len(eps)):
                matrix=edge_matrixes[(eps[i],eps[j])]
                h_stack.append(matrix)
        h_stack=tuple(h_stack)
        stacked.append(np.hstack(h_stack)) # Horizontal Stack
 stacked=tuple(stacked)
 edge_matrix=np.vstack(stacked) # Vertical Stack
 return [edge_matrix,total_edge_matrix,total_edges] # To Return Edge Matrix , Total Edges and Total Edge Matrix

'''
Demonstration of how the above function works:
Input   :    eps=[0,4.0]]

D_12:  np.array([[8,9,7,8,10],
                 [3,5,7,7,9],
                 [9,10,10,3,4]])

Output:
edge_matrix: 
array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 1., 0., 1., 1., 0.],
        [1., 0., 0., 0., 0., 1., 1., 1., 1., 0.],
        [0., 0., 0., 1., 1., 0., 0., 0., 1., 1.]])
       
total_edge_matrix:
 array([[0., 3.],
        [3., 9.]])
        
total_edges:
15.0

'''