import igraph as ig
import plotly.graph_objs as go
import numpy as np
from itertools import combinations
from scipy.sparse.csgraph import connected_components
import plotly.express as px
from tqdm import tqdm

def maximal_graph_from_alpha_complex(my_alpha_complex, top_dimension, a1 = 1):

    maximal_graph = []
    
    for curr_dim in reversed(range(top_dimension + 1)): 
    
        print("Calculating dimension", curr_dim, "of", len(my_alpha_complex[curr_dim]), "facets")
    
        with tqdm( total = len(my_alpha_complex[curr_dim]) ) as pbar:
            
            for simplex, value in my_alpha_complex[curr_dim]:

                if value < a1:
                    accept = True 
                    for accepted_cliques in maximal_graph: 
                        if set(simplex) <= set(accepted_cliques):
                            accept = False
            
                    if accept: 
                        maximal_graph.append(simplex) 
    
                pbar.update(1)

    return maximal_graph

def split_codimension_of_maximal_graph(maximal_graph, top_dimension):

    ambient_dim = max([len(clique) for clique in maximal_graph])

    # First, isolate the top level if it's not the ambient space 
    print("First, isolate the top level if it's not the ambient space ...")
    
    top_level_graph = []
    resolution_graph = []
    
    for clique in maximal_graph: 
        resolution_graph.append(clique) 
    
    while len(resolution_graph) > 0: 
    
        clique = resolution_graph[0]
        
        if len(clique) <= top_dimension + 1 and clique not in top_level_graph: 
            top_level_graph.append(clique)
        else: 
            for subclique in list(combinations(clique, len(clique) - 1)):
                if subclique not in resolution_graph: 
                    resolution_graph.append(subclique)
        
        resolution_graph.pop(0)
    
    print("Second, break cliques into halfs ...")
    new_graph = []

    # with tqdm( total = len(top_level_graph) ) as pbar:
        
    #     for curr_clique in top_level_graph: 
        
    #         if len(curr_clique) == top_dimension + 1: 
        
    #             for idx_start in range(len(curr_clique) - 3):
        
    #                 proposed_new_clique = curr_clique[idx_start:idx_start + 4]
        
    #                 if proposed_new_clique not in new_graph:
    #                     new_graph.append(proposed_new_clique)
        
    #         if len(curr_clique) == top_dimension: 
        
    #             for idx_start in range(len(curr_clique) - 2):
        
    #                 proposed_new_clique = curr_clique[idx_start:idx_start + 3]
        
    #                 if proposed_new_clique not in new_graph:
    #                     new_graph.append(proposed_new_clique)
        
    #         if len(curr_clique) == top_dimension - 1: 
        
    #             for idx_start in range(len(curr_clique) - 1):
        
    #                 proposed_new_clique = curr_clique[idx_start:idx_start + 2]
        
    #                 if proposed_new_clique not in new_graph:
    #                     new_graph.append(proposed_new_clique)

    #         pbar.update(1)

    with tqdm( total = len(top_level_graph) ) as pbar:
        
        for curr_clique in top_level_graph: 
        
            if len(curr_clique) == top_dimension + 1: 
        
                for proposed_new_clique in list(combinations(curr_clique, 4))[:len(curr_clique) - 2]:
                
                    if proposed_new_clique not in new_graph:
                        new_graph.append(proposed_new_clique)
        
            if len(curr_clique) == top_dimension: 
        
                for proposed_new_clique in list(combinations(curr_clique, 3))[:len(curr_clique) - 1]:
                
                    if proposed_new_clique not in new_graph:
                        new_graph.append(proposed_new_clique)
        
            if len(curr_clique) == top_dimension - 1: 
        
                for proposed_new_clique in list(combinations(curr_clique, 2))[:len(curr_clique)]:
                
                    if proposed_new_clique not in new_graph:
                        new_graph.append(proposed_new_clique)

            pbar.update(1)
    
    # Resolve any degenerate cliques that appear in above cliques 
    
    final_graph = []

    print("Resolve any degenerate cliques that appear in above cliques  ...")

    with tqdm( total = len(new_graph)**2 ) as pbar:
    
        for a_clique in new_graph:
            accept = True
            for another_clique in new_graph: 
                if set(a_clique) < set(another_clique):
                    accept = False
                pbar.update(1)
        
            if accept: 
                final_graph.append(a_clique)

    return final_graph

def draw_abstract_simplcial_complex(cplx, D):

    node_names = []
    k = 0
    node_2_idx = {}
    for node, val in cplx[0]:
        node_names.append(node[0])
        node_2_idx[node[0]] = k
        k += 1
    
    max_name = len(node_names)

    edge_list = []
    Y = np.zeros((max_name, max_name))

    for edge, val in cplx[1]: 
        edge_list.append([node_2_idx[edge[0]], node_2_idx[edge[1]]])
        Y[node_2_idx[edge[0]], node_2_idx[edge[1]]] = 1

    groups = connected_components(Y)[1]

    G=ig.Graph(edge_list, directed=False)
    layt=G.layout('kk', dim=3)
    #layt=G.layout('fr3d', dim=3)
    S = np.array(layt)
    N = len(layt)

    fig = go.Figure()
    
    things_to_plot = []
        
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
    
    Xe=[]
    Ye=[]
    Ze=[]
    for e in edge_list:
        Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
        Ye+=[layt[e[0]][1],layt[e[1]][1], None]
        Ze+=[layt[e[0]][2],layt[e[1]][2], None]
    
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
    
    i = []
    j = []
    k = []
    for simplex, val in cplx[3]:
        if len(simplex) == D + 1:
            for idx_1, idx_2, idx_3 in list(combinations(simplex, 3)):
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
            for idx_1, idx_2, idx_3 in list(combinations(simplex, 3)):
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

def draw_clique_graph(E):

    node_names = []
    for clique in E: 
        for node in clique: 
            node_names.append(node) 

    node_names = list(set(node_names))
    max_name = max(node_names)

    edge_list = []
    Y = np.zeros((max_name + 1, max_name + 1))

    for clique in E: 
        for v1 in clique: 
            for v2 in clique: 
                edge_list.append([v1, v2])
                Y[v1, v2] = 1

    groups = connected_components(Y)[1]

    G=ig.Graph(edge_list, directed=False)
    #layt=G.layout('kk', dim=3)
    layt=G.layout('fr3d', dim=3)
    S = np.array(layt)
    N = len(layt)

    fig = go.Figure()
    
    things_to_plot = []
        
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
    
    Xe=[]
    Ye=[]
    Ze=[]
    for e in edge_list:
        Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
        Ye+=[layt[e[0]][1],layt[e[1]][1], None]
        Ze+=[layt[e[0]][2],layt[e[1]][2], None]
    
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
    
    i = []
    j = []
    k = []
    for simplex in E:
        if len(simplex) == 4:
            for idx_1, idx_2, idx_3 in list(combinations(simplex, 3)):
                i.append(idx_1)
                j.append(idx_2)
                k.append(idx_3)
                    
    i = np.array(i)
    j = np.array(j)
    k = np.array(k)
                    
    things_to_plot.append(go.Mesh3d(x=S[:,0], y=S[:,1], z=S[:,2],alphahull=5, opacity=0.4, color='purple', i=i, j=j, k=k,hoverinfo='none'))

    i = []
    j = []
    k = []
    for simplex in E:
        if len(simplex) == 3:
            for idx_1, idx_2, idx_3 in list(combinations(simplex, 3)):
                i.append(idx_1)
                j.append(idx_2)
                k.append(idx_3)
                    
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