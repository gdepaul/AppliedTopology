from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import connected_components
import plotly.graph_objs as go
import itertools
import plotly.express as px
import math 
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

def ordered_voronoi_diagram(X,landmark_points,order):
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = []
    for i in range(len(xx)):
        row_labels = []
        for j in range(len(xx[0])):
            distances = []
            for q in landmark_points:
                distances.append(math.dist((xx[i,j],yy[i,j]), q))

            my_indices = np.argsort(distances)

            label = 0;
            for k in range(order):
                label = 10*label + my_indices[k]

            row_labels.append(label)
        Z.append(row_labels)
        
    Z = np.array(Z)
        
    values = list(set(list(Z.ravel())))
    new_values = range(len(values))
    
    res = {values[i]: new_values[i] for i in range(len(values))}
    for i in range(len(Z)):
        for j in range(len(Z[0])):
            Z[i,j] = res[Z[i,j]]
    

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6), dpi=80)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap='tab20',
           aspect='auto', origin='lower')

    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = landmark_points
    plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def draw_2d_simplicial_complex(simplicial_complex, pos=None, return_pos=False, ax = None):
    """
    Draw a simplicial complex up to dimension 2 from a list of simplices, as in [1].
        
        Args
        ----
        simplicial_complex: a SAGE SimplicialComplex structure. (Added by Greg DePaul)
        
        pos: dict (default=None)
            If passed, this dictionary of positions d:(x,y) is used for placing the 0-simplices.
            The standard nx spring layour is used otherwise.
           
        ax: matplotlib.pyplot.axes (default=None)
        
        return_pos: dict (default=False)
            If True returns the dictionary of positions for the 0-simplices.
            
        References
        ----------    
        .. [1] I. Iacopini, G. Petri, A. Barrat & V. Latora (2019)
               "Simplicial Models of Social Contagion".
               Nature communications, 10(1), 2485.
               
        Authors: 
        .. I. Iacopini (Original Author) - github-username: iaciac
        .. G. DePaul - github-username: gdepaul
    """

    # Obtain simplices from SAGE simplicial complex
    vertices = set()
    for face in simplicial_complex.facets():
        for vertex in list(face):
            vertices.add(vertex)
    vertex_list = list(vertices)
    node_names_to_index = { vertex_list[i]:i for i in range(len(vertex_list))}

    simplices = []
    for face in simplicial_complex.facets():
        new_face = []
        for vertex in list(face):
            new_face.append(node_names_to_index[vertex])
        simplices.append(new_face)
    
    #List of 0-simplices
    nodes =list(set(itertools.chain(*simplices)))
    
    #List of 1-simplices
    edges = list(set(itertools.chain(*[[tuple(sorted((i, j))) for i, j in itertools.combinations(simplex, 2)] for simplex in simplices])))

    #List of 2-simplices
    triangles = list(set(itertools.chain(*[[tuple(sorted((i, j, k))) for i, j, k in itertools.combinations(simplex, 3)] for simplex in simplices])))
    
    if ax is None: ax = plt.gca()
    ax.set_xlim([-1.1, 1.1])      
    ax.set_ylim([-1.1, 1.1])
    ax.get_xaxis().set_ticks([])  
    ax.get_yaxis().set_ticks([])
    ax.axis('off')
       
    if pos is None:
        # Creating a networkx Graph from the edgelist
        G = nx.Graph()
        G.add_edges_from(edges)
        # Creating a dictionary for the position of the nodes
        pos = nx.spring_layout(G)
        
    # Drawing the edges
    for i, j in edges:
        (x0, y0) = pos[i]
        (x1, y1) = pos[j]
        line = plt.Line2D([ x0, x1 ], [y0, y1 ],color = 'black', zorder = 1, lw=0.7)
        ax.add_line(line);
    
    # Filling in the triangles
    for i, j, k in triangles:
        (x0, y0) = pos[i]
        (x1, y1) = pos[j]
        (x2, y2) = pos[k]
        tri = plt.Polygon([ [ x0, y0 ], [ x1, y1 ], [ x2, y2 ] ],
                          edgecolor = 'black', facecolor = plt.cm.Blues(0.6),
                          zorder = 2, alpha=0.4, lw=0.5)
        ax.add_patch(tri);

    # Drawing the nodes 
    for i in nodes:
        (x, y) = pos[i]
        circ = plt.Circle([ x, y ], radius = 0.02, zorder = 3, lw=0.5,
                          edgecolor = 'Black', facecolor = u'#ff7f0e')
        ax.add_patch(circ);

    if return_pos: return pos

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
#         i = []
#         j = []
#         k = []
#         for simplex, val in cplx[3]:
#             if len(simplex) == D + 1:
#                 for idx_1, idx_2, idx_3 in list(itertools.combinations(simplex, 3)):
#                     i.append(node_2_idx[idx_1])
#                     j.append(node_2_idx[idx_2])
#                     k.append(node_2_idx[idx_3])

#         i = np.array(i)
#         j = np.array(j)
#         k = np.array(k)

#         things_to_plot.append(go.Mesh3d(x=S[:,0], y=S[:,1], z=S[:,2],alphahull=5, opacity=0.4, color='purple', i=i, j=j, k=k,hoverinfo='none'))

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

def SublevelSetFiltration(chosen_landmarks, S, max_dimension = -1, max_radius = 1):
    
    n = S.shape[0]
    
    V = defaultdict(list)
    
    assigned_voronoi_cells = np.argsort(cdist(S, chosen_landmarks), axis=1)
    
    for i, curr_point in enumerate(S):
        V[assigned_voronoi_cells[i][0]].append(curr_point)
            
    if max_dimension < 0: 
        max_dimension = len(S[0,:])
        
    VR_complex = defaultdict(list)
    X = SimplexTree()
    
    def simplex_proportion(simplex): 
        
        fuzzy = 1
        
        simplex_set = set(simplex)

        count = 0
        for i in range(n): 
            subface = set(assigned_voronoi_cells[i,0:len(simplex) + fuzzy])
            if simplex_set.issubset(subface):
                count += 1

        if count != 0:
            return 1/count
        else:
            return np.inf
    
    print("Evaluating Dimension", 0)
    
    with tqdm(total = len(chosen_landmarks)) as pbar:
    
        for i, s in enumerate(chosen_landmarks): 
            value = simplex_proportion([i])
            if value < max_radius:
                VR_complex[0].append(([i], value))
                X.add_simplex([i], value)
            pbar.update(1)
    
    print("Evaluating Dimension", 1)
    
    Y = np.zeros((len(chosen_landmarks), len(chosen_landmarks)))
    adjacency = np.zeros((len(chosen_landmarks), len(chosen_landmarks)))
    with tqdm(total = len(chosen_landmarks) ** 2) as pbar:
        for i in range(len(chosen_landmarks)):
            curr_row = []
            
            if X.contains_simplex([i]):

                for j in range(len(chosen_landmarks)): 
                    
                    if X.contains_simplex([j]):

                        center_distance = simplex_proportion([i,j])

                        if center_distance < max_radius:
                            VR_complex[1].append(([i,j], center_distance))
                            Y[i,j] = center_distance
                            adjacency[i,j] = 1
                            X.add_simplex([i,j], center_distance)
                        else:
                            Y[i,j] = math.inf

                    pbar.update(1)
                                                
            else: 
                pbar.update(len(chosen_landmarks))
                
    print("\tNumber of Connected Components: ", connected_components(adjacency)[0])

    for curr_dim in range(2,max_dimension + 1):
        
        print("Estimating Number of Facets for dimension ", curr_dim, "Part 1:")
        
        facets_to_consider = VR_complex[curr_dim-1]
        visited_prev_words = SimplexTree()
        visited_prev_word_list = []
        
        if len(facets_to_consider) == 0:
            print("No facets to consider")
            break
        
        with tqdm(total = len(facets_to_consider)) as pbar:
            for facet, val in facets_to_consider:
                sub_facet = facet[:-1]
                if not visited_prev_words.contains_simplex(sub_facet):
                    visited_prev_words.add_simplex(sub_facet,0.0)
                    visited_prev_word_list.append(sub_facet)
                pbar.update(1)
                    
        print("Estimating Number of Facets for dimension ", curr_dim, "Part 2:")
        
        if len(visited_prev_word_list) == 0:
            print("No facets to consider")
            break
        
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
        
        if len(Sigma) == 0:
            print("No facets to consider")
            break
        
        with tqdm(total = len(Sigma)) as pbar:
            for simplex in Sigma:
      
                value = simplex_proportion(simplex)
                        
                if value != math.inf and value < max_radius:
                    VR_complex[curr_dim].append((simplex, value))
                    X.add_simplex(simplex, value)

                pbar.update(1)
    
    return VR_complex

def mystery_dataset_1():
    
    n = 10000

    Theta = 2 * math.pi * np.random.rand(n)
    V = 2 * math.pi * np.random.rand(n)
    
    a = 10
    b = 5
    epsilon = 0.1
    perturb = 20

    def parametrized_mystery(u,v):

        return ((a + b * math.cos(v)) * math.cos(u) + b / perturb * np.random.normal(), (a + b * math.cos(v)) * math.sin(u) + b / perturb * np.random.normal(), b * math.sin(v) * math.cos( u / 2.0 ) + b / perturb * np.random.normal(), b * math.sin(v) * math.sin( u / 2.0 ) + b / perturb * np.random.normal())

    zfunc = np.vectorize(parametrized_mystery)

    X, Y, Z, T = zfunc(Theta, V)
    
    return np.stack((X,Y,Z,T)).T

def mystery_dataset_2():
    
    n = 10000

    Theta = 2 * math.pi * np.random.rand(n)
    V = 2 * math.pi * np.random.rand(n)
    
    a = 10
    b = 5
    epsilon = 0.1
    perturb = 20

    def parametrized_mystery(u,v):

        x = a * math.cos(u)*math.sin(v)
        y = a * math.sin(u)*math.sin(v)
        z = a * math.cos(v)
    
        return (y * z, x * z, x * y, 2 * x ** 2 + 3 * y ** 2 + 7 * z ** 2)

    zfunc = np.vectorize(parametrized_mystery)

    X, Y, Z, T = zfunc(Theta, V)
    
    return np.stack((X,Y,Z,T)).T