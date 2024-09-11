from collections import defaultdict
import numpy as np
from scipy.sparse.csgraph import connected_components
import plotly.graph_objs as go
import itertools
import plotly.express as px
import math 
from tqdm import tqdm

class SimplexTree: 
    
    def __init__(self): 
        
        self.X = [-1, defaultdict(lambda: [ 0.0, defaultdict(list)] ) ]

    def contains_simplex(self, my_tuple): 
        
        curr_level = self.X
        for index in my_tuple: 
            if index in curr_level[1].keys():
                curr_level = curr_level[1][index] 
            else: 
                return False 
            
        return True
    
    def simplex_val(self, my_tuple): 
        
        curr_level = self.X
        for index in my_tuple: 
            if index in curr_level[1].keys():
                curr_level = curr_level[1][index] 
            else: 
                return math.inf 
            
        return curr_level[0]
    
    def simplex_leaves(self, my_tuple): 
        
        curr_level = self.X
        for index in my_tuple: 
            if index in curr_level[1].keys():
                curr_level = curr_level[1][index] 
            else: 
                return [] 
            
        return list(curr_level[1].keys())
    
    def add_simplex(self, new_simplex,val):
    
        curr_level = self.X
        for index in new_simplex[:-1]: 
            if index in curr_level[1].keys():
                curr_level = curr_level[1][index] 
            else: 
                return False 
        
        curr_level[1][new_simplex[-1]] = [ val, defaultdict(lambda: [ 0.0, defaultdict(list)] ) ]
        
        return True
    
def draw_geometric_simplcial_complex(cplx, S, draw_points = True, draw_edges = True, draw_surfaces = True):

    node_names = []
    k = 0
    node_2_idx = {}
    for node, val in cplx[0]:
        node_names.append(node[0])
        node_2_idx[node[0]] = k
        k += 1
        
    D = max(cplx.keys())
    
    max_name = len(node_names)

    edge_list = []
    Y = np.zeros((max_name, max_name))

    for edge, val in cplx[1]: 
        edge_list.append([node_2_idx[edge[0]], node_2_idx[edge[1]]])
        Y[node_2_idx[edge[0]], node_2_idx[edge[1]]] = 1

    groups = connected_components(Y)[1]

    fig = go.Figure()
    
    things_to_plot = []
        
    if draw_points:    
        things_to_plot.append(
            go.Scatter3d(x=S[:,0], y=S[:,1], z=S[:,2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=groups,                # set color to an array/list of desired values
                    colorscale=px.colors.qualitative.Dark24,   # choose a colorscale
                    opacity=1
                    ), 
                hoverinfo='none',
                showlegend=False)
        )
    
    if draw_edges:
        Xe=[]
        Ye=[]
        Ze=[]
        for e in edge_list:
            Xe+=[S[e[0]][0],S[e[1]][0], None]# x-coordinates of edge ends
            Ye+=[S[e[0]][1],S[e[1]][1], None]
            Ze+=[S[e[0]][2],S[e[1]][2], None]

        things_to_plot.append(
                    go.Scatter3d(x=Xe,
                       y=Ye,
                       z=Ze,
                       mode='lines',
                       line=dict(color='rgb(125,125,125)', width=1),
                       hoverinfo='none',
                       showlegend=False
                    )
            )
    
    if draw_surfaces:
        i = []
        j = []
        k = []
        for simplex, val in cplx[3]:
            if len(simplex) == D + 1:
                for idx_1, idx_2, idx_3 in list(itertools.combinations(simplex, 3)):
                    i.append(node_2_idx[idx_1])
                    j.append(node_2_idx[idx_2])
                    k.append(node_2_idx[idx_3])

        i = np.array(i)
        j = np.array(j)
        k = np.array(k)

        things_to_plot.append(go.Mesh3d(x=S[:,0], y=S[:,1], z=S[:,2],alphahull=5, opacity=0.4, color='purple', i=i, j=j, k=k,hoverinfo='none'))

        i = []
        j = []
        k = []
        for simplex, val in cplx[2]:
            if len(simplex) == D:
                for idx_1, idx_2, idx_3 in list(itertools.combinations(simplex, 3)):
                    i.append(node_2_idx[idx_1])
                    j.append(node_2_idx[idx_2])
                    k.append(node_2_idx[idx_3])

        i = np.array(i)
        j = np.array(j)
        k = np.array(k)

        things_to_plot.append(go.Mesh3d(x=S[:,0], y=S[:,1], z=S[:,2],alphahull=5, opacity=0.4, color='cyan', i=i, j=j, k=k,hoverinfo='none'))
    
    layout = go.Layout(
            autosize=False,
            width=1000,
            height=1000
        )
            
    fig = go.Figure(data=things_to_plot, layout=layout)

    fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )
            
    fig.show()
    
def VietorisRips(S, max_dimension = -1, max_radius = 2): 
    
    if max_dimension < 0: 
        max_dimension = len(S[0,:])
    
    R = np.sqrt(max_radius)
    
    VR_complex = defaultdict(list)
    X = SimplexTree()
    
    for i, s in enumerate(S): 
        VR_complex[0].append(([i], 0.0))
        X.add_simplex([i], 0.0)

    print("Evaluating Dimension", 1)
    
    Y = np.zeros((len(S), len(S)))
    adjacency = np.zeros((len(S), len(S)))
    with tqdm(total = len(S) ** 2) as pbar:
        for i in range(len(S)):
            curr_row = []
            for j in range(len(S)): 
                center_distance = np.linalg.norm(S[i] - S[j])
                
                if center_distance < max_radius:
                    VR_complex[1].append(([i,j], center_distance))
                    Y[i,j] = center_distance
                    adjacency[i,j] = 1
                    X.add_simplex([i,j], center_distance)
                else:
                    Y[i,j] = math.inf

                pbar.update(1)
                
    print("\tNumber of Connected Components: ", connected_components(adjacency)[0])

    for curr_dim in range(2,max_dimension + 1):
        
        print("Estimating Number of Facets for dimension ", curr_dim, "Part 1:")
            
        facets_to_consider = VR_complex[curr_dim-1]
        visited_prev_words = SimplexTree()
        visited_prev_word_list = []
        
        with tqdm(total = len(facets_to_consider)) as pbar:
            for facet, val in facets_to_consider:
                sub_facet = facet[:-1]
                if not visited_prev_words.contains_simplex(sub_facet):
                    visited_prev_words.add_simplex(sub_facet,0.0)
                    visited_prev_word_list.append(sub_facet)
                pbar.update(1)
                    
        print("Estimating Number of Facets for dimension ", curr_dim, "Part 2:")
        
        Sigma = []
        with tqdm(total = len(visited_prev_word_list)) as pbar:
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
                        
                pbar.update(1)
        
        print("Evaluating Dimension", curr_dim)

        with tqdm(total = len(Sigma)) as pbar:
            for simplex in Sigma:
                value = 0
                for subface in itertools.combinations(simplex, len(simplex) - 1):
                    value = max(X.simplex_val(subface), value)

                if value != math.inf:
                    VR_complex[curr_dim].append((simplex, value))
                    X.add_simplex(simplex, value)

                pbar.update(1)
    
    return VR_complex