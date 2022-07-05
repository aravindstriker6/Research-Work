import numpy as np


def same_norm_l2(X_c1_curr, X_c2_curr, epsilon_1, epsilon_2):
    new = np.zeros([X_c1_curr.shape[0], X_c2_curr.shape[0]])
    for i in tqdm(range(X_c1_curr.shape[0])):
        for j in range(X_c2_curr.shape[0]):
            final = np.linalg.norm(X_c1_curr[i] - X_c2_curr[j], ord=2)
            if final <= epsilon_1 + epsilon_2:
                new[i][j] = 1
            else:
                new[i][j] = 0
    return new


def same_norm_l_infinity(X_c1_curr, X_c2_curr, epsilon_1, epsilon_2):
    new = np.zeros([X_c1_curr.shape[0], X_c2_curr.shape[0]])
    for i in tqdm(range(X_c1_curr.shape[0])):
        for j in range(X_c2_curr.shape[0]):
            final = np.linalg.norm(X_c1_curr[i] - X_c2_curr[j], ord=np.inf)
            if final <= epsilon_1 + epsilon_2:
                new[i][j] = 1
            else:
                new[i][j] = 0
    return new


def min_l2(x, epsilon):
    final_sum = 0
    for i in range(len(x)):
        if abs(x[i]) >= epsilon:
            final_sum += (abs(x[i]) - epsilon) ** 2
        else:
            continue
    final_sum = final_sum ** (1 / 2)
    return final_sum


def test_l2_infinity(x, epsilon_2, epsilon_infinity):
    calculate = min_l2(x, epsilon_infinity)
    if calculate <= epsilon_2:
        return float(1)
    else:
        return float(0)


def different_norm(X_c1_curr, X_c2_curr, epsilon_2, epsilon_infinity):
    new = np.zeros([X_c1_curr.shape[0], X_c2_curr.shape[0]])
    for i in (range(X_c1_curr.shape[0])):
        for j in range(X_c2_curr.shape[0]):
            ans = abs(X_c1_curr[i] - X_c2_curr[j])
            new[i][j] = test_l2_infinity(ans, epsilon_2, epsilon_infinity)
    return new


def edge_matrix_calculator(epsilon_user, X_c1_curr, X_c2_curr):
    total_edge_matrix = np.empty([len(epsilon_user), len(epsilon_user)], dtype=float)
    edge_matrixes = {}
    total_edges = 0
    for i in range(tqdm(len(epsilon_user))):
        new = 0
        for j in range(tqdm(len(epsilon_user))):
            if epsilon_user[i][0] == epsilon_user[j][0]:
                if epsilon_user[i][0] == 'l2' and epsilon_user[j][0] == 'l2':
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
                if epsilon_user[i][0] == 'l2':
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

