import numpy as np


def same_norm_l2(X_c1_curr, X_c2_curr):
    final_l2_distance = np.zeros([X_c1_curr.shape[0], X_c2_curr.shape[0]])
    for i in tqdm(range(X_c1_curr.shape[0])):
        for j in range(X_c2_curr.shape[0]):
            final_l2_distance[i][j] = np.linalg.norm(X_c1_curr[i] - X_c2_curr[j], ord=2)
    return final_l2_distance


def same_norm_l_infinity(X_c1_curr, X_c2_curr):
    final_l_infinity_distance = np.zeros([X_c1_curr.shape[0], X_c2_curr.shape[0]])
    for i in tqdm(range(X_c1_curr.shape[0])):
        for j in range(X_c2_curr.shape[0]):
            final_l_infinity_distance[i][j] = np.linalg.norm(X_c1_curr[i] - X_c2_curr[j], ord=np.inf)
    return final_l_infinity_distance


def same_norm_l1(X_c1_curr, X_c2_curr):
    final_l1_distance= np.zeros([X_c1_curr.shape[0], X_c2_curr.shape[0]])
    for i in tqdm(range(X_c1_curr.shape[0])):
        for j in range(X_c2_curr.shape[0]):
            final_l1_distance[i][j] = np.linalg.norm(X_c1_curr[i] - X_c2_curr[j], ord=1)
    return final_l1_distance

def test_l2_infinity(x, epsilon_infinity):
    final_sum = 0
    for i in range(len(x)):
        if abs(x[i]) >= epsilon_infinity:
            final_sum += (abs(x[i]) - epsilon_infinity) ** 2
        else:
            continue
    final_sum = final_sum ** (1 / 2)
    return final_sum

def different_norm_l2_l_infinity(X_c1_curr, X_c2_curr,epsilon_infinity):
    distance_l2_l_infinity = np.zeros([X_c1_curr.shape[0], X_c2_curr.shape[0]])
    for i in (range(X_c1_curr.shape[0])):
        for j in range(X_c2_curr.shape[0]):
            ans = abs(X_c1_curr[i] - X_c2_curr[j])
            distance_l2_l_infinity[i][j] = test_l2_infinity(ans,epsilon_infinity)
    return  distance_l2_l_infinity

def test_l1_infinity(x,epsilon_infinity):
    final_sum=0
    for i in range(len(x)):
        if abs(x[i])>=epsilon_infinity:
            final_sum+=abs(abs(x[i])-epsilon_infinity)
        else:
            continue
    return final_sum

def different_norm_l1_l_infinity(X_c1_curr, X_c2_curr,epsilon_infinity):
    distance_l1_l_infinity= np.zeros([X_c1_curr.shape[0], X_c2_curr.shape[0]])
    for i in (range(X_c1_curr.shape[0])):
        for j in range(X_c2_curr.shape[0]):
            ans = abs(X_c1_curr[i] - X_c2_curr[j])
            distance_l1_l_infinity[i][j] = test_l1_infinity(ans, epsilon_infinity)
    return distance_l1_l_infinity


def find_z(X):
    total = 0
    z = [0]  # Initialize first element as 0
    for i in range(X.shape[0]):
        total = total + X[i]
        z.append(total)
    z = np.array(z)  # Final Z vector
    return z


# Finding Optimal K using Linear Search and finding lambda value
def test_l1_l2(X, epsilon_1):
    X = abs(X)
    X.sort()
    X = X[::-1]
    w = X[0] - epsilon_1
    for k in range(1, X.shape[0]):
        if w < k * X[k]:
            w += X[k]
            continue
        else:
            break
    lambda_value = w / k

    return [X, k, lambda_value]


# Finding Minimum Epsilon 2 that intersects Epsilon 1
def find_epsilon_2(K, ans, lambda_value):
    y = []
    for i in range(0, ans.shape[0]):
        if i < K:
            y.append(lambda_value)
        else:
            y.append(abs(ans[i]))
    y = np.array(y)
    distance = np.linalg.norm(y)
    return distance


# When both the norms are different
def different_norm_11_l2(X_c1_curr, X_c2_curr, epsilon_1):
    distance_l1_l2 = np.zeros([X_c1_curr.shape[0], X_c2_curr.shape[0]])
    for i in (range(X_c1_curr.shape[0])):
        for j in range(X_c2_curr.shape[0]):
            ans = abs(X_c1_curr[i] - X_c2_curr[j])
            X, K, lambda_value = test_l1_l2(ans, epsilon_1)
            distance_l1_l2[i][j]= find_epsilon_2(K, X, lambda_value)
    return distance_l1_l2 


def edge_matrix_calculator(epsilon_user, X_c1_curr, X_c2_curr, storage_same_metrics, storage_different_metrics):
    total_edge_matrix = np.empty([len(epsilon_user), len(epsilon_user)], dtype=float)
    edge_matrixes = {}
    total_edges = 0
    for i in range((len(epsilon_user))):
        for j in range((len(epsilon_user))):
            if epsilon_user[i][0] == epsilon_user[j][0]:
                    epsilon_1=epsilon_user[i][1]
                    epsilon_2=epsilon_user[j][1]
                    key = epsilon_user[i][0]
                    if key in storage_same_metrics:
                        distance_matrix= storage_same_metrics[key]
                    else:
                        if epsilon_user[i][0]=='l1':
                         distance_matrix = same_norm_l1(X_c1_curr, X_c2_curr)
                        elif epsilon_user[i][0]=='l2':
                         distance_matrix = same_norm_l2(X_c1_curr, X_c2_curr)
                        else:
                          distance_matrix = same_norm_l_infinity(X_c1_curr, X_c2_curr) 
                        storage_same_metrics[epsilon_user[i][0]] = distance_matrix
                    new=distance_matrix<=epsilon_1+epsilon_2
                        
            elif (epsilon_user[i][0] == 'l1' and epsilon_user[j][0] == 'l2') or (epsilon_user[i][0] == 'l2' and epsilon_user[j][0] == 'l1') :
                 if epsilon_user[i][0] == 'l1' and epsilon_user[j][0] == 'l2':
                   epsilon_1 = epsilon_user[i][1]
                   epsilon_2 = epsilon_user[j][1]
                 else:
                   epsilon_1 = epsilon_user[j][1]
                   epsilon_2 = epsilon_user[i][1]
              
                 key = ('l1','l2',epsilon_1)
                 if key in storage_different_metrics:
                    distance_matrix= storage_different_metrics[key]
                 else:
                   distance_matrix= different_norm_11_l2(X_c1_curr, X_c2_curr, epsilon_1)
                   key = ('l1','l2',epsilon_1)
                   storage_different_metrics[key] = distance_matrix
                 new=distance_matrix<=epsilon_2
                    
            elif (epsilon_user[i][0] == 'l1' and epsilon_user[j][0] == 'l_infinity') or (epsilon_user[i][0] == 'l_infinity' and epsilon_user[j][0] == 'l1') :
                 if epsilon_user[i][0] == 'l1' and epsilon_user[j][0] == 'l_infinity':
                   epsilon_1 = epsilon_user[i][1]
                   epsilon_infinity = epsilon_user[j][1]
                 else:
                   epsilon_1 = epsilon_user[j][1]
                   epsilon_infinity = epsilon_user[i][1]
              
                 key = ('l1','l_infinity',epsilon_infinity)
                 if key in storage_different_metrics:
                    distance_matrix= storage_different_metrics[key]
                 else:
                   distance_matrix= different_norm_l1_l_infinity(X_c1_curr, X_c2_curr,epsilon_infinity)
                   key =('l1','l_infinity',epsilon_infinity)
                   storage_different_metrics[key] = distance_matrix
                 new=distance_matrix<=epsilon_1
            else:
                 if epsilon_user[i][0] == 'l2' and epsilon_user[j][0] == 'l_infinity':
                   epsilon_2 = epsilon_user[i][1]
                   epsilon_infinity = epsilon_user[j][1]
                 else:
                   epsilon_2 = epsilon_user[j][1]
                   epsilon_infinity = epsilon_user[i][1]
              
                 key = ('l2','l_infinity',epsilon_infinity)
                 if key in storage_different_metrics:
                    distance_matrix= storage_different_metrics[key]
                 else:
                   distance_matrix= different_norm_l1_l_infinity(X_c1_curr, X_c2_curr,epsilon_infinity)
                   key =('l1','l_infinity',epsilon_infinity)
                   storage_different_metrics[key] = distance_matrix
                 new=distance_matrix<=epsilon_2
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
    return [edge_matrix, total_edge_matrix, total_edges, storage_same_metrics,
        storage_different_metrics]  # To Return Edge Matrix , Total Edges and Total Edge Matrix
