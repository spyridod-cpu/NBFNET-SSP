import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

from torchdrug import layers
from torchdrug.layers import functional


class GeneralizedRelationalConv(layers.MessagePassingBase):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=True):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)
        if dependent:
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            self.relation = nn.Embedding(num_relation, input_dim)


    #Message function of the model.
    #input: the graph and the represenations of the nodes.
    def message(self, graph, input):
        assert graph.num_relation == self.num_relation

        batch_size = len(graph.query)
        #get nodes sending and receiving a message
        node_in, node_out, relation = graph.edge_list.t()
        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation, self.input_dim)
        else:
            #get representation of the edges
            relation_input = self.relation.weight.expand(batch_size, -1, -1)
        relation_input = relation_input.transpose(0, 1)
        node_input = input[node_in]
        # get representation of the edges according to relation. We only have relation 0.
        edge_input = relation_input[relation]
        #edge_weight = graph.edge_weight.reshape_as(edge_input)
        #edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        #edge_weight = torch.ones(size=relation_input.size(), device= relation_input.device)
        edge_weight = graph.edge_weight.unsqueeze(-1).unsqueeze(-1).expand_as(edge_input)

        #from the message functions only add is used.
        if self.message_func == "transe":
            message = edge_input + node_input
        elif self.message_func == "distmult":
            message = edge_input * node_input
        # message = reprenentation of edge * edge_weight + representation of node sending the message
        elif self.message_func == "add":
            message = edge_input * edge_weight + node_input
        elif self.message_func == "rotate":
            node_re, node_im = node_input.chunk(2, dim=-1)
            edge_re, edge_im = edge_input.chunk(2, dim=-1)
            message_re = node_re * edge_re - node_im * edge_im
            message_im = node_re * edge_im + node_im * edge_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        #augment each message with boundary condition
        message = torch.cat([message, graph.boundary])

        return message

    #Aggregate function of the model.
    #input: the graph and the messages produced by the message function
    #output: the aggregated messages, the message source nodes for each node
    def aggregate(self, graph, message):
        # get nodes sending and receiving a message
        node_in, node_out = graph.edge_list.t()[:2]
        node_out = graph.edge_list[:, 1]
        #include self loops for each node to compare with the boundary condition
        node_in = torch.cat([node_in, torch.arange(graph.num_node, device=graph.device)])
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])

        #edge weights are not used in min aggregation function.
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        #edge_weight = torch.cat([graph.edge_weight, torch.full((graph.num_node,), fill_value=float(-1), device=self.device)])
        edge_weight = edge_weight.unsqueeze(-1).unsqueeze(-1)
        degree_out = graph.degree_out.unsqueeze(-1).unsqueeze(-1) + 1
        #nodes = torch.Tensor([i for i in range(0, graph.num_node + 1)], device=graph.device).requires_grad_()


        if self.aggregate_func == "sum":
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "mean":
            update = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "max":
            update, argmin = scatter_max(message, node_out, dim=0, dim_size=graph.num_node)
            #message_source = torch.cat((node_in, torch.tensor([graph.num_node], device=graph.device)))
        elif self.aggregate_func == "min":
            #find the minimum message from unaltered messages. Scaling with edge weights is
            #done in the message function
            update, argmin = scatter_min(message, node_out, dim=0, dim_size=graph.num_node)
            #message_source = torch.cat((node_in, torch.tensor([graph.num_node], device=graph.device)))
            #predecessor = nodes[message_source[argmin].long()]
        elif self.aggregate_func == "pna":
            mean = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            sq_mean = scatter_mean(message ** 2 * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            max = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            min = scatter_min(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
            #predecessor=None
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        return update


    #Only eligible for use with cuda kernel. The program runs on cpu, so we call
    #message and aggregate functions seperately.
    def message_and_aggregate(self, graph, input):
        return super(GeneralizedRelationalConv, self).message_and_aggregate(graph, input)
        '''
        if graph.requires_grad or self.message_func == "rotate":
            return super(GeneralizedRelationalConv, self).message_and_aggregate(graph, input)
        '''
        assert graph.num_relation == self.num_relation

        batch_size = len(graph.query)
        input = input.flatten(1)
        boundary = graph.boundary.flatten(1)

        degree_out = graph.degree_out.unsqueeze(-1) + 1
        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation, self.input_dim)
            relation_input = relation_input.transpose(0, 1).flatten(1)
        else:
            relation_input = self.relation.weight.repeat(1, batch_size)
        adjacency = graph.adjacency.transpose(0, 1)

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        if self.aggregate_func == "sum":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = update + boundary
        elif self.aggregate_func == "mean":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = (update + boundary) / degree_out
        elif self.aggregate_func == "max":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            update = torch.max(update, boundary)
        elif self.aggregate_func == "pna":
            sum = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            sq_sum = functional.generalized_rspmm(adjacency, relation_input ** 2, input ** 2, sum="add", mul=mul)
            max = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            min = functional.generalized_rspmm(adjacency, relation_input, input, sum="min", mul=mul)
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            max = torch.max(max, boundary)
            min = torch.min(min, boundary)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        return update.view(len(update), batch_size, -1)

    #Run the output of a layer through a linear layer and an activation function and return the
    #updated node representation
    def combine(self, input, update):
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        #output = update[0]
        return output