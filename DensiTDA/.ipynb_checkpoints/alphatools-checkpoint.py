import numpy as np
from math import comb
import numpy.linalg as LA
from ctypes import * 
import ctypes.util
import plotly.graph_objs as go
from collections import defaultdict
from plotly.offline import iplot
import itertools
from qpsolvers import Problem, solve_problem #solve_qp, 
from tqdm import tqdm
import qpsolvers
import gudhi
import matplotlib.pyplot as plot
from numpy import genfromtxt
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from scipy.sparse.csgraph import connected_components

class SimplexTree: 
    
    def __init__(self): 
        
        self.X = defaultdict(lambda: defaultdict(list))

    def contains_simplex(self, my_tuple): 
        
        curr_level = self.X
        for index in my_tuple: 
            if index in curr_level.keys():
                curr_level = curr_level[index] 
            else: 
                return False 
            
        return True
    
    def simplex_leaves(self, my_tuple): 
        
        curr_level = self.X
        for index in my_tuple: 
            if index in curr_level.keys():
                curr_level = curr_level[index] 
            else: 
                return [] 
            
        return list(curr_level.keys())
    
    def add_simplex(self,new_simplex):
    
        curr_level = self.X
        for index in new_simplex[:-1]: 
            if index in curr_level.keys():
                curr_level = curr_level[index] 
            else: 
                return False 
        
        curr_level[new_simplex[-1]] = defaultdict(lambda: defaultdict(list))
        
        return True

def loop_task_1(batch_i, S, P, R):
    
    Y_skel_partial = []
    for i in batch_i:
        for j in range(len(S)): 
            center_distance = np.linalg.norm(S[i] - S[j])
            if i < j and R[i] + R[j] >= center_distance:
                Y_skel_partial.append([i,j])
                Y_skel_partial.append([j,i])
                                    
    return Y_skel_partial

# def loop_task_1(i, S, P, R):
    
#     Y_skel_partial = []
#     for j in range(len(S)): 
#         center_distance = np.linalg.norm(S[i] - S[j])
#         if i < j and R[i] + R[j] >= center_distance:
#             Y_skel_partial.append([i,j])
#             Y_skel_partial.append([j,i])
                                    
#     return Y_skel_partial

def loop_task_2(batch_landmarks, S, P, Y):

    currNG = []
    currReverse = []
    currA = []
    currV = []
    currF = []

    for landmark in batch_landmarks:


        curr_N_G = [j for j in range(len(S)) if Y[landmark][j] == 1]

        currNG.append(curr_N_G)

        currReverse.append({curr_N_G[i]: i for i in range(len(curr_N_G))})
        
        A_i = lambda i : S[i] - S[landmark]
        V_i = lambda i : 1/2 * (np.linalg.norm(S[i]) ** 2 - np.linalg.norm(S[landmark]) ** 2 - P[i] + P[landmark])
            
        currA.append(np.array([A_i(j) for j in curr_N_G], dtype=c_double))
        currV.append(np.array([V_i(j) for j in curr_N_G], dtype=c_double))
        currF.append(-1 * np.array(S[landmark], dtype=c_double))
    
    return batch_landmarks, currNG, currReverse, currA, currV, currF

# def loop_task_2(batch_landmarks, S, P, Y):

#     indices_of_S = np.arange(len(S))

#     xv, yv = np.meshgrid(indices_of_S, batch_landmarks)

#     currNG = Y == 1

#     A_ij = np.vectorize(lambda i, j : S[i] - S[j])
#     V_ij = np.vectorize(lambda i, j : 1/2 * (np.linalg.norm(S[i]) ** 2 - np.linalg.norm(S[j]) ** 2 - P[i] + P[j]))
#     F_j = np.vectorize(lambda landmark : -1 * np.array(S[landmark], dtype=c_double))

#     currA = A_ij(xv, yv).T
#     currV = V_ij(xv, yv).T
#     currF = F_j(batch_landmarks)
    
#     return batch_landmarks, currNG, currA, currV, currF

def loop_task_3(simplex_indices, Sigma, d, BigN_G, BigN_G2S, BigA, BigV, BigF, S, P, a1, primtol):

    ambient_dim = len(S[0])
    H = np.identity(ambient_dim,dtype=c_double)

    tempX = []
    tempAlpha = []

    for simplex_index in simplex_indices: 
        simplex = Sigma[simplex_index]

        if isinstance(simplex, int):
            landmark = simplex
        else: 
            landmark = simplex[0]

        N_G = BigN_G[landmark]
        idx_2_n_g = BigN_G2S[landmark]
        A = BigA[landmark]
        V = BigV[landmark]
        f = BigF[landmark]

        if not isinstance(simplex, int):
            J = list(simplex)
            J.remove(landmark)
        else: 
            J = []

        not_J = []
        for l in N_G:
            if l not in J:
                not_J.append(l)

        J = [idx_2_n_g[j] for j in J]
        not_J = [idx_2_n_g[nj] for nj in not_J]
                                       
        G = A[not_J,:]
        h = V[not_J]

        A_eq = A[J,:]
        b_eq = V[J]

        problem = Problem(H, f, G, h, A_eq, b_eq)
        solution = solve_problem(problem, solver="daqp")
        y = solution.x
                            
                        
        if y is not None: 

            if d > 0 and LA.norm(G) != 0 and not solution.is_optimal(primtol):
                continue

            fval = 1/2 * np.linalg.norm(y - S[landmark]) ** 2

            objmax = (P[landmark] + a1)/2

            if fval < objmax - primtol:

                curr_weight = 2*fval - P[landmark] 

                if isinstance(simplex, int):
                    simplex = [simplex]

                tempX.append(simplex)
                tempAlpha.append((simplex, curr_weight))

    return tempX, tempAlpha

def compute_weighted_cech_graph(S, P, a1):

    # compute the 1 skeleton of the weighted cech complex <-> just checking if the ball cover intersects pairwise
    R = [np.sqrt(a1 + P[i]) for i in range(len(S))]

    batch_size = max(1000, int(len(S) / 24))
    
    batches = []
    curr_index = 0
    while curr_index + batch_size < len(S): 
        batches.append(range(curr_index, curr_index + batch_size))
        curr_index += batch_size
    batches.append(range(curr_index, len(S)))

    with ProcessPoolExecutor(max_workers=8) as pool:

        #result = list(tqdm(pool.map(loop_task_1,  range(len(S)), repeat(S), repeat(P), repeat(R) ), total = len(S)))
        result = list(tqdm(pool.map(loop_task_1,  batches, repeat(S), repeat(P), repeat(R) ), total = len(batches)))
        
    Y = np.zeros((len(S), len(S)))
    Y_skel = []
    for Y_skel_partial in result: 
        Y_skel += Y_skel_partial

        for curr_tuple in Y_skel_partial: 
            Y[tuple(curr_tuple)] = 1

    return Y_skel, Y

def compute_alpha_complex(S, P, a1, D, primtol=0.000001):

    ambient_dim = len(S[0])
    H = np.identity(ambient_dim,dtype=c_double)
        
    # compute the 1 skeleton of the weighted cech complex <-> just checking if the ball cover intersects pairwise
    R = [np.sqrt(a1 + P[i]) for i in range(len(S))]
    
    print("Generating 1-Dimensional Weighted Cech Complex")
        
    batch_size = max(1000, int(len(S) / 24))
    
    batches = []
    curr_index = 0
    while curr_index + batch_size < len(S): 
        batches.append(range(curr_index, curr_index + batch_size))
        curr_index += batch_size
    batches.append(range(curr_index, len(S)))

    with ProcessPoolExecutor(max_workers=8) as pool:

        #result = list(tqdm(pool.map(loop_task_1,  range(len(S)), repeat(S), repeat(P), repeat(R) ), total = len(S)))
        result = list(tqdm(pool.map(loop_task_1,  batches, repeat(S), repeat(P), repeat(R) ), total = len(batches)))
        
    Y = np.zeros((len(S), len(S)))
    Y_skel = []
    for Y_skel_partial in result: 
        Y_skel += Y_skel_partial

        for curr_tuple in Y_skel_partial: 
            Y[tuple(curr_tuple)] = 1
    
    # with tqdm(total = len(S) ** 2) as pbar:
        
    #     Y = []
    #     Y_skel = []
    #     for i in range(len(S)): 
    #         curr_row = []
    #         for j in range(len(S)): 
    #             center_distance = np.linalg.norm(S[i] - S[j])
    #             if R[i] + R[j] >= center_distance:
    #                 Y_skel.append([i,j])
    #                 curr_row.append(1)
    #             else: 
    #                 curr_row.append(0) 
                    
    #             pbar.update(1)
                
    #         Y.append(curr_row)
    
    print("\tTotal Edges of Cech Graph: ", int(len(Y_skel) / 2) )
    
    def max_degree_of_cech_graph():
        
        record_instances = defaultdict(int)
        
        for edge in Y_skel: 
            record_instances[edge[0]] += 1
            
        max_val = 0
        for k, v in record_instances.items():
            if v > max_val: 
                max_val = v 
                
        return max_val
        
    print("\tHighest Degree of Cech Graph: ", max_degree_of_cech_graph())

    def avg_degree_of_cech_graph():
        
        record_instances = defaultdict(int)
        
        for edge in Y_skel: 
            record_instances[edge[0]] += 1
            
        max_val = 0
        for k, v in record_instances.items():
            max_val += v 
                
        return max_val / len(S)
        
    print("\tAverage Degree of Cech Graph: ", avg_degree_of_cech_graph())

    print("\tNumber of Connected Components: ", connected_components(Y)[0])
        
    print("Begin Computing Alpha Complex")
     
    alpha_complex = defaultdict(list)
    N = len(S) + 1
    d_1 = D + 1
        
    X = SimplexTree()

    # Preprocess Neighbors
    print("Preprocessing Dual Matrices: ", len(S))

    batch_size = 1000 #max(1000, int(len(S) / 24))

    if len(S) <= batch_size: 
        BigN_G = []
        BigN_G2S = []
        BigA = []
        BigV = []
        BigF = []
    
        with tqdm( total = len(S) ) as pbar:
            
            for landmark in range(len(S)):

                curr_N_G = [j for j in range(len(S)) if Y[landmark][j] == 1]
                BigN_G.append(curr_N_G)

                BigN_G2S.append({curr_N_G[i]: i for i in range(len(curr_N_G))})
        
                A_i = lambda i : S[i] - S[landmark]
                V_i = lambda i : 1/2 * (np.linalg.norm(S[i]) ** 2 - np.linalg.norm(S[landmark]) ** 2 - P[i] + P[landmark])
            
                BigA.append(np.array([A_i(j) for j in curr_N_G], dtype=c_double))
                BigV.append(np.array([V_i(j) for j in curr_N_G], dtype=c_double))
                BigF.append(-1 * np.array(S[landmark], dtype=c_double))
    
                pbar.update(1)
    else:
        BigN_G = len(S) * [0]
        BigN_G2S = len(S) * [0]
        BigA = len(S) * [0]
        BigV = len(S) * [0]
        BigF = len(S) * [0]
    
        batches = []
        curr_index = 0
        while curr_index + batch_size < len(S): 
            batches.append(range(curr_index, curr_index + batch_size))
            curr_index += batch_size
        batches.append(range(curr_index, len(S)))
        
        with ProcessPoolExecutor(max_workers=8) as pool:
                            
            result = list(tqdm(pool.map(loop_task_2,  batches, repeat(S), repeat(P), repeat(Y) ), total = len(batches)))
    
        for batch_landmarks, currNG, currReverse, currA, currV, currF in result:
            for k, landmark in enumerate(batch_landmarks): 
                BigN_G[landmark] = currNG[k]
                BigN_G2S[landmark] = currReverse[k]
                BigA[landmark] = currA[k]
                BigV[landmark] = currV[k]
                BigF[landmark] = currF[k]
                
    for d in range(D + 1): 

        print("*********** BEGIN DIMENSION " + str(d) + " ***********")
        
        Sigma = []
        
        if d == 0: 
            Sigma = list(range(len(S)))
        
        if d == 1: 
            Sigma = []
            for sigma in Y_skel: 
                if sigma[0] < sigma[1]:
                    Sigma.append(sigma)

        if d > 1: 
            
            print("Estimating Number of Facets for dimension ", d)
            
            facets_to_consider = alpha_complex[d-1]
            visited_prev_words = SimplexTree()
            visited_prev_word_list = []
            
            for facet, val in facets_to_consider:
                sub_facet = facet[:-1]
                if not visited_prev_words.contains_simplex(sub_facet):
                    visited_prev_words.add_simplex(sub_facet)
                    visited_prev_word_list.append(sub_facet)
                    
            Sigma = [] 
            for word in visited_prev_word_list:
                indices = X.simplex_leaves(word)
                for choose_pair in itertools.combinations(indices, r = 2):
                    suggested_word = word + list(choose_pair)
                    flag = True
                    for subsimplex in list(itertools.combinations(suggested_word, len(suggested_word) - 1)):
                        if not X.contains_simplex(subsimplex): 
                            flag = False
                            break

                    if flag:
                        Sigma.append(word + list(choose_pair))
                    
        print("\tPossible Facets: ", len(Sigma))

        if len(Sigma) > 0:

            # if len(Sigma) < 500000:
            #     batch_size = 500000
            # else: 
            #     batch_size = int(len(Sigma) / 24)

            #batch_size = 10000
            batch_size = 100000

            
            if len(Sigma) < batch_size:
            
                with tqdm(total=len(Sigma)) as pbar:
        
                    for simplex in Sigma: 
        
                        if isinstance(simplex, int):
                            landmark = simplex
                        else: 
                            landmark = simplex[0]
        
                        N_G = BigN_G[landmark]

                        idx_2_n_g = BigN_G2S[landmark]
        
                        A = BigA[landmark]
                        V = BigV[landmark]
                        f = BigF[landmark]

                        #print(simplex, A @ A.T) 
                        #print(S[landmark].shape)
                        #print(simplex, A @ S[landmark] - V.T)
        
                        if not isinstance(simplex, int):
                            J = list(simplex)
                            J.remove(landmark)
                        else: 
                            J = []
        
                        not_J = []
                        for l in N_G:
                            if l != landmark and l not in J:
                                not_J.append(l)
                        
                        J = [idx_2_n_g[j] for j in J]
                        not_J = [idx_2_n_g[nj] for nj in not_J]

                        if len(not_J) > 0:
                            G = A[not_J,:]
                            h = V[not_J]
                        else: 
                            G = np.array([0],dtype=c_double)
                            h = np.array([0],dtype=c_double)

                        if len(not_J) > 0:
                            A_eq = A[J,:]
                            b_eq = V[J]
                        else: 
                            A_eq = np.array([0],dtype=c_double)
                            b_eq = np.array([0],dtype=c_double)
        
                        problem = Problem(H, f, G, h, A_eq, b_eq)
                        solution = solve_problem(problem, solver="daqp")
                        y = solution.x
                            
                        
                        if y is not None: 

                            if d > 0 and LA.norm(G) != 0 and not solution.is_optimal(primtol):
                                pbar.update(1)
                                continue
        
                            fval = 1/2 * np.linalg.norm(y - S[landmark]) ** 2
        
                            objmax = (P[landmark] + a1)/2
        
                            if fval < objmax - primtol:
        
                                curr_weight = 2*fval - P[landmark]
        
                                if isinstance(simplex, int):
                                    simplex = [simplex]
        
                                if not X.contains_simplex(simplex):
    
                                    X.add_simplex(simplex)
                                    alpha_complex[d].append((simplex, curr_weight))
        
                        pbar.update(1)
            else:
                batches = []
                curr_index = 0
                while curr_index + batch_size < len(Sigma): 
                    batches.append(range(curr_index, curr_index + batch_size))
                    curr_index += batch_size
                batches.append(range(curr_index, len(Sigma)))
                
                with ProcessPoolExecutor(max_workers=8) as pool:
                                    
                    result = list(tqdm(pool.map(loop_task_3,  batches, repeat(Sigma), repeat(d), repeat(BigN_G), repeat(BigN_G2S), repeat(BigA), repeat(BigV), repeat(BigF), repeat(S), repeat(P), repeat(a1), repeat(primtol)), total = len(batches)))
                
                for tempX, tempAlpha in result:
                    for simplex, curr_tuple in zip(tempX, tempAlpha):
                        if not X.contains_simplex(simplex): 
    
                            X.add_simplex(simplex)
                            alpha_complex[d].append(curr_tuple)
    
        print("\tFinal Number of Facets: ", len(alpha_complex[d]))
        
    return alpha_complex

def draw_alpha_complex_2d(alpha_complex, S, p, a1): 

    R = [np.sqrt(a1 + p[i]) for i in range(len(S))]

    Y = []
    for i in range(len(S)): 
        curr_row = []
        for j in range(len(S)): 
            center_distance = np.linalg.norm(S[i] - S[j])
            if R[i] + R[j] >= center_distance and i != j:
                curr_row.append(1)
            else: 
                curr_row.append(0) 
        Y.append(curr_row)

    fig = go.Figure()
    
    # Set axes properties
    fig.update_xaxes(range=[-5, 5], zeroline=False)
    fig.update_yaxes(range=[-2, 8])

    for i in range(len(S)):
        center = S[i]
        center_x = center[0]
        center_y = center[1]
        
        fig.add_shape(type="circle",
            xref="x", yref="y",
            fillcolor="PaleTurquoise",
            x0=center_x - R[i], y0 = center_y - R[i], x1=center_x + R[i], y1=center_y + R[i],
            line_color="LightSeaGreen",
            opacity=0.7
        )
        
    fig.add_trace(go.Scatter(x=S[:,0], y=S[:,1], mode="markers"))

    for i in range(len(S)):
        first_center = S[i]
        first_center_x = first_center[0]
        first_center_y = first_center[1]
        for j in range(len(S)):
            second_center = S[j]
            second_center_x = second_center[0]
            second_center_y = second_center[1]
            
            if i != j and Y[i][j] == 1:
                fig.add_shape(
                    dict(type="line", x0=first_center_x, x1=second_center_x, y0=first_center_y, y1=second_center_y, line_color="black")
                )

    if len(alpha_complex[2]) > 0:
        for simplex, val in alpha_complex[2]:
                    
            x0, y0 = S[simplex[0]]
            x1, y1 = S[simplex[1]]
            x2, y2 = S[simplex[2]]
                    
            fig.add_trace(
                go.Scatter(x=[x0,x1,x2,x0], y=[y0,y1,y2,y0], fill="toself", opacity=0.5)
            )

    # Set figure size
    fig.update_layout(width=800, height=800)

    fig.show()


def draw_alpha_complex(alpha_complex, S, a1, points_only = True):
    
    fig = go.Figure()
    
    things_to_plot = []
    
    # Set axes properties
    fig.update_xaxes(range=[-5, 5], zeroline=False)
    fig.update_yaxes(range=[-2, 8])

    filtered_nodes = []
    
    for xyz, simplex in zip(S, alpha_complex[0]):
        
        if simplex[1] < a1: 
            filtered_nodes.append(xyz)
            
    filtered_nodes = np.array(filtered_nodes)

    if points_only: 
        
        things_to_plot.append(go.Scatter3d(x=filtered_nodes[:,0], y=filtered_nodes[:,1], z=filtered_nodes[:,2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=filtered_nodes[:,2],                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=1
                ), showlegend=False))
    else: 
    
        for simplex, weight in alpha_complex[1]:
            if weight < a1:
                things_to_plot.append(go.Scatter3d(x=[S[simplex[0]][0], S[simplex[1]][0]], y=[S[simplex[0]][1], S[simplex[1]][1]], z=[S[simplex[0]][2], S[simplex[1]][2]], mode='lines',line=dict(
                                                    color="black",
                                                    width=10),showlegend=False))
            
        for simplex, weight in alpha_complex[2]:
            if weight < a1:
                X = [S[simplex[0]][0], S[simplex[1]][0], S[simplex[2]][0]]
                Y = [S[simplex[0]][1], S[simplex[1]][1], S[simplex[2]][1]]
                Z = [S[simplex[0]][2], S[simplex[1]][2], S[simplex[2]][2]]
                
                i = np.array([0])
                j = np.array([1])
                k = np.array([2])
                
                things_to_plot.append(go.Mesh3d(x=X, y=Y, z=Z,alphahull=5, opacity=0.4, color='cyan', i=i, j=j, k=k))
        
    layout = go.Layout(
        autosize=False,
        width=1000,
        height=1000
    )
        
    fig = go.Figure(data=things_to_plot, layout=layout)
        
    fig.show()