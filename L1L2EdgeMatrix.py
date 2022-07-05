import numpy as np
#Finding Z vector
def find_z(X):
    total=0
    z=[0]  # Initialize first element as 0
    for i in range(X.shape[0]):
        total=total+X[i]
        z.append(total)
    z=np.array(z) # Final Z vector
    return z

#Finding Optimal K using Linear Search and finding lambda value
def test_l1_l2(X,epsilon_1,epsilon2):
      X=abs(X)
      X.sort()
      X=X[::-1]
      w=X[0]-epsilon_1
      for k in range(1,X.shape[0]):
        if w<k*X[k]:
            w+=X[k]
            continue
        else:
            break
      lambda_value=w/k
      
      return [X,k,lambda_value]
#Finding Minimum Epsilon 2 that intersects Epsilon 1
def find_epsilon_2(K,ans,lambda_value):
    y=[]
    for i in range(0,ans.shape[0]):
        if i<K:
            y.append(lambda_value)
        else:
            y.append(abs(ans[i]))
    y=np.array(y)
    distance=np.linalg.norm(y)
    return distance


# When both the norms are different
def different_norm(X_c1_curr,X_c2_curr,epsilon_1,epsilon2):
  new=np.zeros([X_c1_curr.shape[0],X_c2_curr.shape[0]])
  for i in (range(X_c1_curr.shape[0])):
        for j in range(X_c2_curr.shape[0]):
            ans=abs(X_c1_curr[i]-X_c2_curr[j])
            X,K,lambda_value=test_l1_l2(ans,epsilon_1,epsilon2)
            min_epsilon_2=find_epsilon_2(K,X,lambda_value)
            if epsilon2>=min_epsilon_2:
               new[i][j]=1
            else:
                new[i][j]=0
                
  return new
# when both norms are epsilon1
def same_norm_l2(X_c1_curr,X_c2_curr,epsilon_1,epsilon_2):
    new=np.zeros([X_c1_curr.shape[0],X_c2_curr.shape[0]])
    for i in tqdm(range(X_c1_curr.shape[0])):
     for j in range(X_c2_curr.shape[0]):
        final=np.linalg.norm(X_c1_curr[i]-X_c2_curr[j])
        if final<=epsilon_1+epsilon_2:
                new[i][j]=1
        else:
                new[i][j]=0
    return new
# when both norms are epsilon2
def same_norm_l1(X_c1_curr,X_c2_curr,epsilon_1,epsilon_2):
    new=np.zeros([X_c1_curr.shape[0],X_c2_curr.shape[0]])
    for i in tqdm(range(X_c1_curr.shape[0])):
     for j in range(X_c2_curr.shape[0]):
        final=np.linalg.norm(X_c1_curr[i]-X_c2_curr[j],ord=1)
        if final<=epsilon_1+epsilon_2:
                new[i][j]=1
        else:
                new[i][j]=0
    return new

def edge_matrix_calculator(epsilon_user, X_c1_curr, X_c2_curr):
    total_edge_matrix = np.empty([len(epsilon_user), len(epsilon_user)], dtype=float)
    edge_matrixes = {}
    total_edges = 0
    for i in range(tqdm(len(epsilon_user))):
        new = 0
        for j in range(tqdm(len(epsilon_user))):
            if epsilon_user[i][0] == epsilon_user[j][0]:
                if epsilon_user[i][0] == 'l1' and epsilon_user[j][0] == 'l1':
                    new = same_norm_l1(X_c1_curr, X_c2_curr, epsilon_user[i][1], epsilon_user[j][1])
                    new = new.astype(float)
                    edge_matrixes[i, j] = new
                    total_edge_matrix[i][j] = new.sum()
                    total_edges += new.sum()
                else:
                    new = same_norm_l_infinity(X_c1_curr, X_c2_curr, epsilon_user[i][1], epsilon_user[j][1])
                    new = new.astype(float)
                    edge_matrixes[i, j] = new
                    total_edge_matrix[i][j] = new.sum()
                    total_edges += new.sum()

            else:
                if epsilon_user[i][0] == 'l1':
                    epsilon_2 = epsilon_user[i][1]
                    epsilon_infinity = epsilon_user[j][1]
                else:
                    epsilon_2 = epsilon_user[j][1]
                    epsilon_infinity = epsilon_user[i][1]
                new = different_norm(X_c1_curr, X_c2_curr, epsilon_2, epsilon_infinity)
                new = new.astype(float)
                edge_matrixes[i, j] = new
                total_edge_matrix[i][j] = new.sum()
                total_edges += new.sum()
    stacked = []
    for i in range(len(epsilon_user)):
        h_stack = []
        for j in range(len(epsilon_user)):
            matrix = edge_matrixes[i, j]
            h_stack.append(matrix)
        h_stack = tuple(h_stack)
        stacked.append(np.hstack(h_stack))  # Horizontal Stack
    stacked = tuple(stacked)
    edge_matrix = np.vstack(stacked)  # Vertical Stack
    return [edge_matrix, total_edge_matrix, total_edges]  # To Return Edge Matrix , Total Edges and Total Edge Matrix

