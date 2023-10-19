import os
import sys
import csv
import osmnx as ox
import torch
import networkx as nx
import matplotlib.pyplot as plt
import yaml
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import util



#Function that produces a random networkx graph.
#input: name of the dataset, path to save
def produce_random_graph(name, path):
    G = nx.graph_atlas(100)
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.savefig("Graph.png", format="PNG")
    edges = G.edges
    res = [list(ele) for ele in edges]
    nodes = len(G.nodes)
    nodes = [[i, 0] for i in range(0,len(G.nodes))]
    weights = torch.randint(1, 20, size=(len(G.edges),1)).tolist()

    with open(path + '/data/' + name + '_edges.txt', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f,delimiter='\t')
        # write multiple rows
        writer.writerows(res)
    with open(path + '/data/' + name + '_nodes.txt', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f,delimiter='\t')
        # write multiple rows
        writer.writerows(nodes)

    with open(path + '/data/' + name + '_weights.txt', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # write multiple rows
        writer.writerows(weights)


#Function that returns two dictionaries, one that maps original ids to [0, num_nodes) and
#in reverse
def graphdict(nodes):
    node_vocab = dict()
    inv_node_vocab = dict()
    i = 0
    for node in nodes:
        node_vocab[node] = i
        inv_node_vocab[i] = node
        i = i+1
    return node_vocab, inv_node_vocab

#Function that produces a road graph given the name of the dataset, the path to the data, and the
#bounding box of the area as defined by Open Street Map.
def produce_road_graph(name, **bounding_box):

    #Get graph G from bounding box in the yaml configuration file.
    G = ox.graph_from_bbox(bounding_box['yhigh'], bounding_box['ylow'], bounding_box['xlow'], bounding_box['xhigh'], network_type='drive')

    ox.plot_graph(G, save=True, bgcolor='w', node_color='black', filepath=os.path.join(os.path.dirname(__file__), '..',
                                                                                       'figures',  name + '_graph.png'))

    nodes = G.nodes
    #Translate original node ids to [0, num_nodes-1] and back
    nodes_vocab, inv_node_vocab = graphdict(nodes)
    node_features = [[node['x'], node['y'], 0] for node in G.nodes._nodes.values()]
    #produce a sequence of ids and 0 as class of each node for nodes.txt
    nodes = [[nodes_vocab[node], 0] for node in nodes]
    weights = []
    #store weights of graph to weights.txt
    for i, j in G.edges():
        weight = G[i][j][0]['length']
        weights.append([weight])

    #store non self looping edges in edges.txt
    edge_list = [[nodes_vocab[edge[0]], nodes_vocab[edge[1]]] for edge in G.edges if nodes_vocab[edge[0]] != nodes_vocab[edge[1]] ]

    #store the dictionaries mapping node ids in two files, nodes_vocab.txt and inv_nodes_vocab.txt
    resultList = new_list = list(map(list, nodes_vocab.items()))
    resultList2 = new_list = list(map(list, inv_node_vocab.items()))

    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', name+ '_nodes.txt')), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f,delimiter='\t')
        # write multiple rows
        writer.writerows(nodes)
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', name + '_edges.txt')), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # write multiple rows
        writer.writerows(edge_list)

    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', name + '_nodes_vocab.txt')), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # write multiple rows
        writer.writerows(resultList)
    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', name + '_inv_nodes_vocab.txt')), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # write multiple rows
        writer.writerows(resultList2)

    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', name + '_weights.txt')), 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        # write multiple rows
        writer.writerows(weights)


if __name__ == '__main__':

    #load yaml configuration file
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    #produce the road graph and store it in text files
    produce_road_graph(cfg.name, **cfg.bounding_box)




