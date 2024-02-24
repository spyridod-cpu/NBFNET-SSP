import csv
import os
import sys
import pprint

import networkx as nx
import torch
from networkx import path_weight
from osmnx import plot_graph_route

from torchdrug import core
from torchdrug.utils import comm
import osmnx as ox
import matplotlib.lines as mlines


sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from nbfnet import dataset, layer, model, task, util








#function that plots and saves paths in figures folder.
def plot_paths(G, model_path, path, folder, t):
    #get graph from bounding box
    G = ox.graph_from_bbox(bounding_box['yhigh'], bounding_box['ylow'], bounding_box['xlow'], bounding_box['xhigh'],
                           network_type='drive')
    G_projected = ox.project_graph(G)
    path_length = path_weight(G, model_path, weight = 'length')

    #plot model path
    fig, ax = plot_graph_route(G_projected, model_path, route_color='b', route_linewidth=1, route_alpha= 0.5, orig_dest_size = 1, node_color = 'black', node_size = 5)
    ax.set_facecolor("white")
    blue_line = mlines.Line2D([], [], color='blue',
                              markersize=15, label='Model Path')
    red_line = mlines.Line2D([], [], color='red',
                             markersize=15, label='Djikstra Path')
    ax.legend(handles=[blue_line, red_line])

    orig = model_path[0]
    dest = model_path[len(model_path) - 1]
    #path = ox.distance.shortest_path(G, orig, dest, weight='length', cpus=1)
    x = (G_projected.nodes[path[0]]["x"], G_projected.nodes[path[-1]]["x"])
    y = (G_projected.nodes[path[0]]["y"], G_projected.nodes[path[-1]]["y"])
    o_marker = ax.scatter(x[0],y[0], marker = 'o', label = 'Source Node')
    x_marker = ax.scatter(x[1], y[1], marker = 'x', label='destination')
    o_line = mlines.Line2D([], [], color='blue', marker='o',
                              markersize=15, label='Source Node')
    x_line = mlines.Line2D([], [], color='blue', marker='x',
                              markersize=15, label='Destination')
    ax.legend(handles = [blue_line,red_line,o_marker, x_marker])

    #path_length_d = path_weight(G_projected, path, weight='length')
    #plot shortest path
    fig2, ax2 = plot_graph_route(G_projected, path, route_linewidth=2, route_alpha= 0.5, orig_dest_size = 0, ax = ax)
    filename = "fig_" + t
    fig2.savefig(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures', filename)))

#read dictionary mapping original ids to the range [0, num_nodes)
def read_vocab(name):
    node_vocab_keys = []
    node_vocab_values = []
    inv_node_vocab_keys = []
    inv_node_vocab_values = []


    with open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', name + '_nodes_vocab.txt')), 'r') as file:
        csv_reader = csv.reader(file, delimiter='\t')
        for node in csv_reader:
            node_vocab_keys.append(int(node[0]))
            node_vocab_values.append(int(node[1]))
            inv_node_vocab_keys.append(int(node[1]))
            inv_node_vocab_values.append(int(node[0]))
        node_vocab = dict(zip(node_vocab_keys, node_vocab_values))
        inv_node_vocab = dict(zip(inv_node_vocab_keys, inv_node_vocab_values))

    return node_vocab, inv_node_vocab



if __name__ == "__main__":
    #parse arguments
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    #working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    logger.warning("Config file: %s" % args.config)
    logger.warning(pprint.pformat(cfg))


    #load dataset
    dataset = core.Configurable.load_config_dict(cfg.dataset)
    #build solver
    solver = util.build_solver(cfg, dataset)
    folder = cfg.dataset.path
    solver.model.eval()
    graph = solver.model.train_graph
    orig = 12
    dest = 20
    mean_distance = 0
    l = 0
    t = 1
    bounding_box = cfg.data.bounding_box
    threshold = cfg.data.threshold
    #get graph from bounding box
    G = ox.graph_from_bbox(bounding_box['yhigh'], bounding_box['ylow'], bounding_box['xlow'], bounding_box['xhigh'], network_type='drive')
    weights = solver.model.train_graph.edge_weight
    G_di = nx.DiGraph(G)
    G_di_normal_weights = nx.DiGraph(G)
    weights = graph.edge_weight.tolist()

    #add normalized weights
    for i, j in G_di.edges():
        G_di[i][j]['weight'] = weights[l]
        l = l + 1

    n = 0
    node_vocab, inv_node_vocab = read_vocab(cfg.data['name'])
    for node in solver.test_set:
        for k in range(0, 10):
            #run model and get model path
            orig = node['node_index']
            dest = k
            try:
                shortest_path = nx.shortest_path(G_di_normal_weights, inv_node_vocab[orig], inv_node_vocab[dest],
                                                 weight='length')
            except:
                continue
            path = solver.model.model.visualize(graph, torch.tensor(orig).unsqueeze(-1), torch.tensor(dest).unsqueeze(-1), torch.tensor(0).unsqueeze(-1))
            path = [node for node in path]
            path_translated = [inv_node_vocab[node] for node in path]
            #if a path exists in the graph
            try:

                path_length = path_weight(G_di_normal_weights, path_translated, weight='length')
            except:
                continue
            #translate path from range[0, num_nodes) to original ids
            path_translated_shortest = [node_vocab[node] for node in shortest_path]

            print(orig == path[0])
            path_length_shortest = path_weight(G_di_normal_weights, shortest_path, weight='length')
            diff = path_length - path_length_shortest
            mean_distance += path_length - path_length_shortest
            n += 1
            G_or = G
            if path_length-path_length_shortest >= threshold:
                plot_paths(G, path_translated, shortest_path, folder, t=str(t))
                f = open(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'figures', 'distances')), "a")
                f.write(str(path_length-path_length_shortest)+"\n")
                f.close()
                t += 1
    mean_distance = mean_distance/n
    print(mean_distance, n)








