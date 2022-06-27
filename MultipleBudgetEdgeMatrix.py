import numpy as np

def build_edge_matrix(D_12,eps):
 total_edge_matrix={}
 total_edges=[]
 for m in range(len(eps)):
         no_edge_matrix=np.empty([len(eps[m]),len(eps[m])], dtype=float)
         edge_matrixes={}
         for i in range(len(eps[m])):
             new=0
             for j in range(len(eps[m])):
                 new= D_12 <= float(eps[m][i])+float(eps[m][j]) #Finding if there is a edge or not
                 new = new.astype(float)
                 no_edge_matrix[i][j]=new.sum() # Calculating total number of edges
                 edge_matrixes[(eps[m][i],eps[m][j])]=new
         total_edge_matrix[tuple(eps[m])]=no_edge_matrix # Storing Number of Edges Matrix
         stacked=[]
         for i in range(len(eps[m])):
                h_stack=[]
                for j in range(len(eps[m])):
                    matrix=edge_matrixes[(eps[m][i],eps[m][j])]
                    h_stack.append(matrix)
                h_stack=tuple(h_stack)
                stacked.append(np.hstack(h_stack))
         stacked=tuple(stacked)
         edge_matrix=np.vstack(stacked) # Final Edge Matrix
         num_edges = len(np.where(edge_matrix!=0)[0])
         total_edges.append((eps[m],num_edges))
 return [edge_matrix,total_edge_matrix,total_edges]

'''
Demonstration of how the above function works:
Input   :    eps=[[0,2.0,0],[0,3.0,1],[0,4.0,2],[0,5.0,0],[0,6.0,0],[0,7.0,0],[0,8.0,0],[0,9.0,0],[0,10,0]]

Output:
edge_matrix: 
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]])
       
total_edge_matrix:
{(0,2.0,0): array([[0., 0., 0.],
        [0., 4., 0.],
        [0., 0., 0.]]),
        
(0,3.0,1): array([[   0.,    0.,    0.],
        [   0., 1260.,    4.],
        [   0.,    4.,    0.]]),
        
(0,4.0,2): array([[0.00000e+00, 4.00000e+00, 0.00000e+00],
        [4.00000e+00, 1.13839e+05, 1.26000e+03],
        [0.00000e+00, 1.26000e+03, 4.00000e+00]]),
        
(0,5.0,0): array([[0.000000e+00, 1.080000e+02, 0.000000e+00],
        [1.080000e+02, 2.049952e+06, 1.080000e+02],
        [0.000000e+00, 1.080000e+02, 0.000000e+00]]),
        
(0,6.0,0): array([[0.000000e+00, 1.260000e+03, 0.000000e+00],
        [1.260000e+03, 5.728112e+06, 1.260000e+03],
        [0.000000e+00, 1.260000e+03, 0.000000e+00]]),
        
(0,7.0,0): array([[      0.,   13333.,       0.],
        [  13333., 6248564.,   13333.],
        [      0.,   13333.,       0.]]),
        
(0,8.0,0): array([[      0.,  113839.,       0.],
        [ 113839., 6250000.,  113839.],
        [      0.,  113839.,       0.]]),
        
(0,9.0,0): array([[      0.,  616129.,       0.],
        [ 616129., 6250000.,  616129.],
        [      0.,  616129.,       0.]]),
        
(0,10, 0): array([[      0., 2049952.,       0.],
        [2049952., 6250000., 2049952.],
        [      0., 2049952.,       0.]])}
        
total_edges:
[([0, 2.0, 0], 4),
 ([0, 3.0, 1], 1268),
 ([0, 4.0, 2], 116371),
 ([0, 5.0, 0], 2050384),
 ([0, 6.0, 0], 5733152),
 ([0, 7.0, 0], 6301896),
 ([0, 8.0, 0], 6705356),
 ([0, 9.0, 0], 8714516),
 ([0, 10, 0], 14449808)]

'''