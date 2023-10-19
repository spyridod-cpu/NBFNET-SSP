from collections.abc import Sequence

import torch
from torch import nn
from torch import autograd

from torch_scatter import scatter_add

from torchdrug import core, layers
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from . import layer


@R.register("model.NBFNet")
class NeuralBellmanFordNetwork(nn.Module, core.Configurable):

    def __init__(self, input_dim, hidden_dims, num_relation=None, symmetric=False,
                 message_func="distmult", aggregate_func="pna", short_cut=False, layer_norm=False, activation="relu",
                 concat_hidden=False, num_mlp_layer=2, dependent=True, remove_one_hop=False,
                 num_beam=10, path_topk=10, num_nodes=1, num_iterations=1, recover = False):
        super(NeuralBellmanFordNetwork, self).__init__()

        if not isinstance(hidden_dims, Sequence):
            hidden_dims = [hidden_dims]
        if num_relation is None:
            double_relation = 1
        else:
            num_relation = int(num_relation)
            double_relation = num_relation * 2
        self.dims = [input_dim] + list(hidden_dims)
        self.num_relation = num_relation
        self.symmetric = symmetric
        self.short_cut = short_cut
        self.concat_hidden = concat_hidden
        self.remove_one_hop = remove_one_hop
        self.num_beam = num_beam
        self.path_topk = path_topk
        self.num_iterations = num_iterations
        self.recover = recover

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(layer.GeneralizedRelationalConv(self.dims[i], self.dims[i + 1], double_relation,
                                                               self.dims[0], message_func, aggregate_func, layer_norm,
                                                               activation, dependent))

        feature_dim = hidden_dims[-1] * (len(hidden_dims) if concat_hidden else 1) + input_dim
        self.query = nn.Embedding(double_relation, input_dim)
        self.mlp = layers.MLP(feature_dim, [feature_dim] * (num_mlp_layer - 1) + [num_nodes])

    def remove_easy_edges(self, graph, h_index, t_index, r_index=None):
        if self.remove_one_hop:
            h_index_ext = torch.cat([h_index, t_index], dim=-1)
            t_index_ext = torch.cat([t_index, h_index], dim=-1)
            if r_index is not None:
                any = -torch.ones_like(h_index_ext)
                pattern = torch.stack([h_index_ext, t_index_ext, any], dim=-1)
            else:
                pattern = torch.stack([h_index_ext, t_index_ext], dim=-1)
        else:
            if r_index is not None:
                pattern = torch.stack([h_index, t_index, r_index], dim=-1)
            else:
                pattern = torch.stack([h_index, t_index], dim=-1)
        pattern = pattern.flatten(0, -2)
        edge_index = graph.match(pattern)[0]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)

    def negative_sample_to_tail(self, h_index, t_index, r_index):
        # convert p(h | t, r) to p(t' | h', r')
        # h' = t, r' = r^{-1}, t' = h
        is_t_neg = (h_index == h_index[:, [0]]).all(dim=-1, keepdim=True)
        new_h_index = torch.where(is_t_neg, h_index, t_index)
        new_t_index = torch.where(is_t_neg, t_index, h_index)
        new_r_index = torch.where(is_t_neg, r_index, r_index + self.num_relation)
        return new_h_index, new_t_index, new_r_index

    def as_relational_graph(self, graph, self_loop=False):
        # add self loop
        # convert homogeneous graphs to knowledge graphs with 1 relation
        edge_list = graph.edge_list
        edge_weight = graph.edge_weight
        if self_loop:
            node_in = node_out = torch.arange(graph.num_node, device=self.device)
            loop = torch.stack([node_in, node_out], dim=-1)
            edge_list = torch.cat([edge_list, loop])
            edge_weight = torch.cat([edge_weight, torch.ones(graph.num_node, device=self.device)])
        relation = torch.zeros(len(edge_list), 1, dtype=torch.long, device=self.device)
        edge_list = torch.cat([edge_list, relation], dim=-1)
        graph = type(graph)(edge_list, edge_weight=edge_weight, num_node=graph.num_node,
                            num_relation=1, meta_dict=graph.meta_dict, **graph.data_dict)
        return graph


    #Bellman ford algorithm.
    #input: The graph, the source node and the type of relation (0).
    def bellmanford(self, graph, h_index, r_index, separate_grad=False):
        #get representation of an edge of relation type 0
        query = self.query(r_index)
        index = h_index.unsqueeze(-1).expand_as(query)
        #set boundary condition as 100 (equivalent to infinity).
        boundary = torch.full((graph.num_node, *query.shape), fill_value=float(100), device=self.device)
        #set boundary condition of source node to 0.
        boundary[h_index.item()][0] = 0
        #boundary.scatter_add_(0, index.unsqueeze(0), query.unsqueeze(0))
        with graph.graph():
            graph.query = query
        with graph.node():
            graph.boundary = boundary

        hiddens = []
        step_graphs = []
        layer_input = boundary
        for i in range(0, self.num_iterations):
            for layer in self.layers:
                #store graph for each layer
                if separate_grad:
                    step_graph = graph.clone().requires_grad_()
                else:
                    step_graph = graph
                #pass through a layer and extract reprentations of each layer
                #and message sources for each node
                hidden, predecessor = layer(step_graph, layer_input)
                if self.short_cut and hidden.shape == layer_input.shape:
                    hidden = hidden + layer_input
                hiddens.append(hidden)
                step_graphs.append(step_graph)
                layer_input = hidden

        node_query = query.expand(graph.num_node, -1, -1)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
        else:
            #each representation of the node is augmented with the representation of the source node,
            #to differentiate representations based on different source nodes.
            output = torch.cat([hiddens[-1], node_query], dim=-1)

        return {
            "node_feature": output,
            "step_graphs": step_graphs,
            "predecessor": predecessor
        }


    #Forward pass through the model.
    #input: The graph and the source node, the type of relation (0).
    def forward(self, graph, h_index, r_index=None, all_loss=None, metric=None):

        shape = h_index.shape
        #homogeneous graph as graph with only one relation
        graph = self.as_relational_graph(graph)
        h_index = h_index.view(-1, 1)
        #t_index = t_index.view(-1, 1)
        r_index = torch.zeros_like(h_index)

        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        #execute bellman ford algorithm
        output = self.bellmanford(graph, h_index[:, 0], r_index[:, 0])
        #extract node representations
        feature = output["node_feature"].transpose(0, 1)
        #extract source nodes for every message
        predecessor = output["predecessor"]
        #index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        #feature = feature.gather(1, index)

        #classify nodes based on node representaions
        score = self.mlp(feature).squeeze(-1)
        return score, predecessor
    '''
    def forward(self, graph, h_index, t_index, r_index=None, all_loss=None, metric=None):
        if all_loss is not None:
            graph = self.remove_easy_edges(graph, h_index, t_index, r_index)

        shape = h_index.shape
        if graph.num_relation:
            graph = graph.undirected(add_inverse=True)
            h_index, t_index, r_index = self.negative_sample_to_tail(h_index, t_index, r_index)
        else:
            graph = self.as_relational_graph(graph)
            h_index = h_index.view(-1, 1)
            t_index = t_index.view(-1, 1)
            r_index = torch.zeros_like(h_index)

        assert (h_index[:, [0]] == h_index).all()
        assert (r_index[:, [0]] == r_index).all()
        output = self.bellmanford(graph, h_index[:, 0], r_index[:, 0])
        feature = output["node_feature"].transpose(0, 1)
        index = t_index.unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(1, index)

        if self.symmetric:
            assert (t_index[:, [0]] == t_index).all()
            output = self.bellmanford(graph, t_index[:, 0], r_index[:, 0])
            inv_feature = output["node_feature"].transpose(0, 1)
            index = h_index.unsqueeze(-1).expand(-1, -1, inv_feature.shape[-1])
            inv_feature = inv_feature.gather(1, index)
            feature = (feature + inv_feature) / 2

        score = self.mlp(feature).squeeze(-1)
        return score.view(shape)
    '''
    '''
    def visualize(self, graph, h_index, t_index, r_index):
        assert h_index.numel() == 1 and h_index.ndim == 1
        graph = graph.undirected(add_inverse=True)

        output = self.bellmanford(graph, h_index, r_index, separate_grad=True)
        feature = output["node_feature"]
        step_graphs = output["step_graphs"]

        index = t_index.unsqueeze(0).unsqueeze(-1).expand(-1, -1, feature.shape[-1])
        feature = feature.gather(0, index).squeeze(0)
        score = self.mlp(feature).squeeze(-1)

        edge_weights = [graph.edge_weight for graph in step_graphs]
        edge_grads = autograd.grad(score, edge_weights)
        for graph, edge_grad in zip(step_graphs, edge_grads):
            with graph.edge():
                graph.edge_grad = edge_grad
        distances, back_edges = self.beam_search_distance(step_graphs, h_index, t_index, self.num_beam)
        paths, weights = self.topk_average_length(distances, back_edges, t_index, self.path_topk)

        return paths, weights

    '''
    '''
    def visualize(self, graph, h_index, t_index, r_index):
        assert h_index.numel() == 1 and h_index.ndim == 1
        output, dec = self.forward(graph, h_index, r_index)
        preds = []
        pred = output
        prev_node2 = -1
        for i in range(0, graph.num_node):
            prev = (graph.edge_list[:, 1] == i).nonzero(as_tuple=False).flatten()
            prev = graph.edge_list[prev, 0]
            #pred[i] = pred[i, prev]
            if not prev.tolist() == []:
                preds.append(prev.tolist())
            else:
                preds.append([i])

        temp_index = t_index.item()
        prev_node = 100
        path = [temp_index]
        prev_index = 0
        if not temp_index == h_index:
            for i in range(0, len(preds)):
                prev_node = torch.Tensor(preds[temp_index]).long()
                for prev_ in prev_node:
                    if prev_ == h_index.item():
                        prev_node2 = h_index.item()
                        break
                    if prev_ in path:
                        if len(prev_node) !=1:
                            prev_node = prev_node[prev_node != prev_]
                        else:
                            prev_index +=1
                            prev_node = torch.Tensor([path[len(path)-1-prev_index]]).long()
                            prev_index += 1
                    else:
                        prev_index = 0
                if prev_node2 != h_index.item():
                    prev_node = prev_node[torch.argmax(pred[0, temp_index, prev_node])].tolist()
                else:
                    path.append(prev_node2)
                    break

                path.append(prev_node)

                temp_index = prev_node
        else:
            path = [h_index.item()]
        path.reverse()
        path = list(dict.fromkeys(path))
        #path.append(t_index.item())
        return path
    
    
    def visualize(self, graph, h_index, t_index, r_index):
        if self.recover:
            assert h_index.numel() == 1 and h_index.ndim == 1
            output, dec = self.forward(graph, h_index, r_index)
            preds = []
            pred = output
            prev_node2 = -1
            for i in range(0, graph.num_node):
                prev = (graph.edge_list[:, 1] == i).nonzero(as_tuple=False).flatten()
                prev = graph.edge_list[prev, 0]
                #pred[i] = pred[i, prev]
                if not prev.tolist() == []:
                    preds.append(prev.tolist())
                else:
                    preds.append([i])
    
            temp_index = t_index.item()
            prev_node = 100
            path = [temp_index]
            if not temp_index == h_index:
                for i in range(0, len(preds)):
                    prev_node = torch.Tensor(preds[temp_index]).long()
                    for prev_ in prev_node:
                        if prev_ == h_index.item():
                            prev_node2 = h_index.item()
                            break
                        if prev_ in path:
                            if len(prev_node) !=1:
                                prev_node = prev_node[prev_node != prev_]
                    if prev_node2 != h_index.item():
                        prev_node = prev_node[torch.argmax(pred[0, temp_index, prev_node])].tolist()
                    else:
                        path.append(prev_node2)
                        break
    
                    path.append(prev_node)
    
                    temp_index = prev_node
            else:
                path = [h_index.item()]
            path.reverse()
            #path.append(t_index.item())
            return path
        else:
                assert h_index.numel() == 1 and h_index.ndim == 1
                output, dec = self.forward(graph, h_index, r_index)
                preds = []
                pred = output
                for i in range(0, graph.num_node):
                    prev = (graph.edge_list[:, 1] == i).nonzero(as_tuple=False).flatten()
                    prev = graph.edge_list[prev, 0]
                    # pred[i] = pred[i, prev]
                    if not prev.tolist() == []:
                        preds.append(prev[torch.argmax(pred[0, i, prev])].tolist())
                    else:
                        preds.append(i)

                temp_index = t_index.item()
                prev_node = 100
                path = [temp_index]
                if not temp_index == h_index:
                    for i in range(0, len(preds)):
                        prev_node = preds[temp_index]
                        if prev_node == h_index.item():
                            path.append(prev_node)
                            break

                        path.append(prev_node)

                        temp_index = prev_node
                else:
                    path = [h_index.item()]
                path.reverse()
                # path.append(t_index.item())
                return path

    '''

    def visualize(self, graph, h_index, t_index, r_index):
        #error recovery algorithm is used
        if self.recover:
            assert h_index.numel() == 1 and h_index.ndim == 1
            #make a forward pass through the model
            output, dec = self.forward(graph, h_index, r_index)
            preds = []
            pred = output
            prev_node2 = -1
            temp_index2 = t_index.item()
            visited = []
            pop = False
            #for every node assemble predictions of predecessor nodes. Only eligible nodes are allowed
            #i.e. nodes that connect to query node.
            for i in range(0, graph.num_node):
                #all eligible predecessor nodes u must be connect to node i with an
                #edge e = (u, i).
                prev = (graph.edge_list[:, 1] == i).nonzero(as_tuple=False).flatten()
                #get the previous nodes u from the list of edges
                prev = graph.edge_list[prev, 0]
                # pred[i] = pred[i, prev]
                #if some node has no predecessors, it is a predecessor to itself.
                if not prev.tolist() == []:
                    preds.append(prev.tolist())
                else:
                    preds.append([i])

            temp_index = t_index.item()
            prev_node = 100
            path = [temp_index]
            visited = path.copy()
            #if we have not reached the source node.
            if not temp_index == h_index:
                #for 2*|V| iterations, guaranteeing convergence.
                for i in range(0, 2*len(preds)):
                    #list of eligible predecessor nodes.
                    prev_node = torch.Tensor(preds[temp_index]).long()
                    #remove the previous visited node from eligible predecessor nodes.
                    # to avoid endless loops.
                    if len(prev_node) != 1 and temp_index2 in prev_node:
                        prev_node = prev_node[prev_node != temp_index2]
                    #mark nodes in the list of eligible predecessor nodes
                    #that have been visited before as ineligible.
                    for prev_ in prev_node:
                        if prev_ == h_index.item():
                            prev_node2 = h_index.item()
                            break
                        if prev_ in visited:
                            if len(prev_node) != 1:
                                prev_node = prev_node[prev_node != prev_]
                                pop = False
                            #if there is only one choice and the node is ineligible,
                            #backtrack until a node with at least one eligible choice is
                            #found.
                            else:
                                pop = True
                                path.pop(len(path)-1)
                    #if currently not backtracking.
                    if prev_node2 != h_index.item() and not pop:
                        #find predecessor node with highest probability.
                        prev_node = prev_node[torch.argmax(pred[0, temp_index, prev_node])].tolist()
                        path.append(prev_node)
                        #mark as visited.
                        visited.append(prev_node)
                        #keep node two steps back from current node.
                        temp_index2 = temp_index
                        #keep predecessor of current node.
                        temp_index = prev_node
                    #if the source node is reached, break.
                    elif prev_node2 == h_index.item():
                        path.append(prev_node2)
                        break
                    #if backtracking.
                    elif pop:
                        #if all of nodes in the path have not been removed.
                        if not path == []:
                            #indexes are reset to the current node.
                            temp_index2 = path[len(path)-1]
                            temp_index = temp_index2
                        else:
                            #readd the destination in the path.
                            path.append(t_index.item())
                            temp_index2 = t_index.item()

                    #path.append(prev_node)

            else:
                path = [h_index.item()]
            #the path is in the form [dest, u_n, u_n-1,..., src], reverse to obtain true path.
            path.reverse()
            return path
        else:
            #same as before but accept loops and dead ends. The runtime is at most |V|
            assert h_index.numel() == 1 and h_index.ndim == 1
            output, dec = self.forward(graph, h_index, r_index)
            preds = []
            pred = output
            for i in range(0, graph.num_node):
                prev = (graph.edge_list[:, 1] == i).nonzero(as_tuple=False).flatten()
                prev = graph.edge_list[prev, 0]
                # pred[i] = pred[i, prev]
                if not prev.tolist() == []:
                    preds.append(prev[torch.argmax(pred[0, i, prev])].tolist())
                else:
                    preds.append(i)

            temp_index = t_index.item()
            prev_node = 100
            path = [temp_index]
            if not temp_index == h_index:
                for i in range(0, len(preds)):
                    prev_node = preds[temp_index]
                    if prev_node == h_index.item():
                        path.append(prev_node)
                        break

                    path.append(prev_node)

                    temp_index = prev_node
            else:
                path = [h_index.item()]
            path.reverse()
            # path.append(t_index.item())
            return path
    '''
    def visualize(self, graph, h_index, t_index, r_index):
        assert h_index.numel() == 1 and h_index.ndim == 1
        output, dec = self.forward(graph, h_index, r_index)
        preds = []
        pred = output
        for i in range(0, graph.num_node):
            prev = (graph.edge_list[:, 1] == i).nonzero(as_tuple=False).flatten()
            prev = graph.edge_list[prev, 0]
            #pred[i] = pred[i, prev]
            if not prev.tolist() == []:
                preds.append(prev[torch.argmax(pred[0, i, prev])].tolist())
            else:
                preds.append(i)

        temp_index = t_index.item()
        prev_node = 100
        path = [temp_index]
        if not temp_index == h_index:
            for i in range(0, len(preds)):
                prev_node = preds[temp_index]
                if prev_node == h_index.item():
                    path.append(prev_node)
                    break

                path.append(prev_node)

                temp_index = prev_node
        else:
            path = [h_index.item()]
        path.reverse()
        #path.append(t_index.item())
        return path
    '''
    '''
    def visualize(self, graph, h_index, t_index, r_index):
        assert h_index.numel() == 1 and h_index.ndim == 1
        output = self.forward(graph, h_index, r_index)

        pred = output.squeeze(0)
        ar = torch.argmax(pred, 1)
        pred_sorted, indices_sorted = torch.sort(pred, 0)
        for i in range(0, len(pred)):
            if i != h_index.item():
                for k in range(0, len(pred) - 1):
                    if [ar[i], i] not in graph.edge_list.tolist():
                        ar[i] = indices_sorted[i][k]
                    else:
                        break
            else:
                ar[i] = i
        temp_index = t_index
        prev_node = 100
        path = []
        for i in range(0, len(pred)):

            prev_node = ar[temp_index]
            if prev_node == temp_index:
                break

            path.append(prev_node)

            temp_index = prev_node
        path.reverse()
        path.append(t_index)
        return path
    '''
        



    @torch.no_grad()
    def beam_search_distance(self, graphs, h_index, t_index, num_beam=10):
        num_node = graphs[0].num_node
        input = torch.full((num_node, num_beam), float("-inf"), device=self.device)
        input[h_index, 0] = 0

        distances = []
        back_edges = []
        for graph in graphs:
            graph = graph.edge_mask(graph.edge_list[:, 0] != t_index)
            node_in, node_out = graph.edge_list.t()[:2]

            message = input[node_in] + graph.edge_grad.unsqueeze(-1)
            msg_source = graph.edge_list.unsqueeze(1).expand(-1, num_beam, -1)

            is_duplicate = torch.isclose(message.unsqueeze(-1), message.unsqueeze(-2)) & \
                           (msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)).all(dim=-1)
            is_duplicate = is_duplicate.float() - \
                           torch.arange(num_beam, dtype=torch.float, device=self.device) / (num_beam + 1)
            # pick the first occurrence as the previous state
            prev_rank = is_duplicate.argmax(dim=-1, keepdim=True)
            msg_source = torch.cat([msg_source, prev_rank], dim=-1)

            node_out, order = node_out.sort()
            node_out_set = torch.unique(node_out)
            # sort message w.r.t. node_out
            message = message[order].flatten()
            msg_source = msg_source[order].flatten(0, -2)
            size = scatter_add(torch.ones_like(node_out), node_out, dim_size=num_node)
            msg2out = torch.repeat_interleave(size[node_out_set] * num_beam)
            # deduplicate
            is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
            is_duplicate = torch.cat([torch.zeros(1, dtype=torch.bool, device=self.device), is_duplicate])
            message = message[~is_duplicate]
            msg_source = msg_source[~is_duplicate]
            msg2out = msg2out[~is_duplicate]
            size = scatter_add(torch.ones_like(msg2out), msg2out, dim_size=len(node_out_set))

            if not torch.isinf(message).all():
                distance, rel_index = functional.variadic_topk(message, size, k=num_beam)
                abs_index = rel_index + (size.cumsum(0) - size).unsqueeze(-1)
                back_edge = msg_source[abs_index]
                distance = distance.view(len(node_out_set), num_beam)
                back_edge = back_edge.view(len(node_out_set), num_beam, 4)
                distance = scatter_add(distance, node_out_set, dim=0, dim_size=num_node)
                back_edge = scatter_add(back_edge, node_out_set, dim=0, dim_size=num_node)
            else:
                distance = torch.full((num_node, num_beam), float("-inf"), device=self.device)
                back_edge = torch.zeros(num_node, num_beam, 4, dtype=torch.long, device=self.device)

            distances.append(distance)
            back_edges.append(back_edge)
            input = distance

        return distances, back_edges

    def topk_average_length(self, distances, back_edges, t_index, k=10):
        paths = []
        average_lengths = []

        for i in range(len(distances)):
            distance, order = distances[i][t_index].flatten(0, -1).sort(descending=True)
            back_edge = back_edges[i][t_index].flatten(0, -2)[order]
            for d, (h, t, r, prev_rank) in zip(distance[:k].tolist(), back_edge[:k].tolist()):
                if d == float("-inf"):
                    break
                path = [(h, t, r)]
                for j in range(i - 1, -1, -1):
                    h, t, r, prev_rank = back_edges[j][h, prev_rank].tolist()
                    path.append((h, t, r))
                paths.append(path[::-1])
                average_lengths.append(d / len(path))

        if paths:
            average_lengths, paths = zip(*sorted(zip(average_lengths, paths), reverse=True)[:k])

        return paths, average_lengths