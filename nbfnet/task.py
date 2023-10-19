import math
import networkx as nx
from networkx import shortest_paths as sp
from torchdrug import data
import torch
from torch.nn import functional as F
from torch.utils import data as torch_data

from ogb import linkproppred

from torchdrug import core, tasks, metrics
from torchdrug.layers import functional
from torchdrug.core import Registry as R


Evaluator = core.make_configurable(linkproppred.Evaluator)
Evaluator = R.register("ogb.linkproppred.Evaluator")(Evaluator)
setattr(linkproppred, "Evaluator", Evaluator)




#Predecessor Prediction task. Dijkstra is run to obtain the ground truth
#predecessors of each node, with respect to a source node. Then, for
#every source node the predictions are treated like a batch and a loss is
#calculated.
@R.register("tasks.PredecessorPrediction")
class PredecessorPrediction(tasks.NodePropertyPrediction, core.Configurable):
    def __init__(self, model, criterion="bce", metric=("accuracy"), num_class=2, verbose=0):
        super().__init__(model, criterion, metric, num_class, verbose)
        self.train_pred = []
        self.valid_pred = []
        self.test_pred = []
        self.train_sources = []
        self.valid_sources = []
        self.test_sources = []
        self.split = "test"

    def preprocess(self, train_set, valid_set, test_set):
        if isinstance(train_set, torch_data.Subset):
            dataset = train_set.dataset
        else:
            dataset = train_set

        dataset.graph = data.Graph(dataset.graph.edge_list, dataset.edge_weights, node_label=dataset.graph.node_label)
        graph = dataset.graph
        self.num_class = graph.num_node
        #setup the NetworkX graph on which the shortest path are calculated
        G = nx.DiGraph()
        weights = graph.edge_weight.tolist()
        G.add_edges_from(graph.edge_list.tolist())
        l = 0
        for i, j in G.edges():
            G[i][j]['weight'] = weights[l]
            l = l + 1

        train_preds = []
        train_sources = []
        valid_preds = []
        valid_sources = []
        test_preds = []
        test_sources = []

        #for every node in the train, valid and test sets, find distances to all other nodes and return
        #the predecessor node as a label.
        for node in train_set.indices:
            train_sources.append(node)
            pred, distance = sp.dijkstra_predecessor_and_distance(G, node, cutoff=None, weight='weight')
            train_preds.append(pred)
        for node in valid_set.indices:
            valid_sources.append(node)
            pred, distance = sp.dijkstra_predecessor_and_distance(G, node, cutoff=None, weight='weight')
            valid_preds.append(pred)
        for node in test_set.indices:
            test_sources.append(node)
            pred, distance = sp.dijkstra_predecessor_and_distance(G, node, cutoff=None, weight='weight')
            test_preds.append(pred)


        self.register_buffer("train_graph", dataset.graph)
        self.register_buffer("valid_graph", dataset.graph)
        self.register_buffer("test_graph", dataset.graph)
        self.train_pred = train_preds
        self.valid_pred = valid_preds
        self.test_pred = test_preds
        self.train_sources = train_sources
        self.valid_sources = valid_sources
        self.test_sources = test_sources

    #function that runs the model, obtains the predictions, makes the targets (predecessor for each node) and
    #returns the predictions and the ground truth
    #input: the source node
    #output: the predictions and the ground truths for each node
    def predict_and_target(self, batch, all_loss=None, metric=None):
        batch_size = len(batch)
        node_0 = batch['node_index']
        #run through the model
        pred, decessor = self.model(self.train_graph, node_0, all_loss, metric=None)
        pred = pred.squeeze(0)

        if self.split == "train":
            node = self.train_sources.index(node_0)
            indices = [i for i in self.train_pred[node].keys()]
            target = torch.zeros(len(pred))
            k = 0
            for i in range(0, len(pred)):
                # if a node does not have a predecessor node
                # the node is a predecessor to itself
                if k not in self.train_pred[node].keys():
                    target[k] = k
                elif k in self.train_pred[node].keys():
                    if self.train_pred[node][k] ==[]:
                        target[k] = k
                    else:
                        target[k] = self.train_pred[node][k][0]
                k+=1

        elif self.split == "valid":
            node = self.valid_sources.index(node_0)
            indices = [i for i in self.train_pred[node].keys()]
            target = torch.zeros(len(pred))
            k = 0
            for i in range(0, len(pred)):
                if k not in self.valid_pred[node].keys():
                    target[k] = k
                elif k in self.valid_pred[node].keys():
                    if self.valid_pred[node][k] ==[]:
                        target[k] = k
                    else:
                        target[k] = self.valid_pred[node][k][0]
                k+=1
        elif self.split == "test":
            node = self.test_sources.index(node_0)
            indices = [i for i in self.train_pred[node].keys()]
            target = torch.zeros(len(pred))
            k = 0
            for i in range(0,len(pred)):
                if k not in self.test_pred[node].keys():
                    target[k] = k
                elif k in self.test_pred[node].keys():
                    if self.test_pred[node][k] ==[]:
                        target[k] = k
                    else:
                        target[k] = self.test_pred[node][k][0]
                k+=1

        return pred, target

    #A forward pass through the model. Obtains the predictions and the ground truths for each node
    #and calculates the loss (cross_entropy).
    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}
        #make predictions
        pred, target = self.predict_and_target(batch, all_loss, metric)


        #calculate loss
        for criterion, weight in self.criterion.items():
            if criterion == "mse":
                loss = F.mse_loss(pred, target, reduction="none")
            elif criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            elif criterion == "ce":
                loss = F.cross_entropy(pred, target.long(), reduction="none")
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)

            name = tasks._get_criterion_name(criterion)

            #loss = loss.squeeze(-1)
            loss = loss.mean()
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric

    #evaluate the model using accuracy as a metric.
    #input: Predictions and ground truths
    #output: accuracy
    def evaluate(self, pred, target):

        #pred = pred.flatten()
        target = target

        metric = {}
        #calculate metric, accuracy is used.
        for _metric in self.metric:
            if _metric == "auroc":
                score = metrics.area_under_roc(pred, target)
            elif _metric == "ap":
                score = metrics.area_under_prc(pred, target)
            elif _metric == "accuracy":
                score = metrics.accuracy(pred, target)
            else:
                raise ValueError("Unknown metric `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric







