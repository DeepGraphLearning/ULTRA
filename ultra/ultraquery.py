import copy
import torch
from torch import nn
from torch.nn import functional as F

from ultra.query_utils import Stack, spmm_max
from ultra.tasks import build_relation_graph, edge_match
from ultra.base_nbfnet import index_to_mask
from torch_geometric.data import Data,Batch


class UltraQuery(nn.Module):
    """
    Query executor for answering multi-hop logical queries.

    Parameters:
        model (nn.Module): GNN model (Ultra) that returns a distribution of scores over nodes
        logic (str, optional): which fuzzy logic system to use, ``godel``, ``product`` or ``lukasiewicz``
        dropout_ratio (float, optional): ratio for traversal dropout
        threshold (float, optional): a score threshold for inductive models pre-trained only on 1-hop link prediction
        more_dropout (float, optional): even more edge dropout because we like to regularize (who doesn't?)
    """

    stack_size = 2

    def __init__(self, model, logic="product", dropout_ratio=0.25, threshold=0.0, more_dropout=0.0):
        super(UltraQuery, self).__init__()
        self.model = RelationProjection(model, threshold)
        self.symbolic_model = SymbolicTraversal()
        self.logic = logic
        self.dropout_ratio = dropout_ratio
        self.more_dropout = more_dropout

    def traversal_dropout(self, graph, h_prob, r_index):
        """Dropout edges that can be directly traversed to create an incomplete graph."""
        sample, h_index = h_prob.nonzero().t()
        r_index = r_index[sample]

        # p1: find all tails
        direct_ei = torch.vstack([graph.edge_index[0], graph.edge_type])
        direct_query = torch.vstack([h_index, r_index])
        direct_mask = edge_match(direct_ei, direct_query)[0]
        # p2: find heads with inverses
        # CAUTION: in some datasets, inverse edge type is rel+1, in some it is rel + num_rel/2
        inverse_ei = torch.vstack([graph.edge_type, graph.edge_index[1]])
        inverse_rel_plus_one = getattr(graph, 'inverse_rel_plus_one', False)
        if inverse_rel_plus_one:
            inverse_r_index = r_index ^ 1
        else:
            inverse_r_index = torch.where(r_index >= graph.num_relations // 2, r_index - graph.num_relations // 2, r_index + graph.num_relations // 2)
        inv_query = torch.vstack([inverse_r_index, h_index])
        inverse_mask = edge_match(inverse_ei, inv_query)[0]

        edge_index = torch.cat([direct_mask, inverse_mask])

        # don't remove edges that break the graph into separate connected components
        h_index, t_index = graph.edge_index
        degree_h = h_index.bincount()
        degree_t = t_index.bincount()
        h_index, t_index = graph.edge_index[:, edge_index]
        must_keep = (degree_h[h_index] <= 1) | (degree_t[t_index] <= 1)
        edge_index = edge_index[~must_keep]

        is_sampled = torch.rand(len(edge_index), device=graph.edge_index.device) <= self.dropout_ratio
        edge_index = edge_index[is_sampled]

        if self.more_dropout > 0.0:
            # More general edge dropout
            more_drop_mask = torch.rand(graph.edge_index.shape[1], device=graph.edge_index.device) <= self.more_dropout
            more_drop_edges = more_drop_mask.nonzero().squeeze(1)
            h_index, t_index = graph.edge_index[:, more_drop_edges]  # maybe add the main edge_index here as well
            must_keep = (degree_h[h_index] <= 1) | (degree_t[t_index] <= 1)
            more_drop_edges = more_drop_edges[~must_keep]
            # Add to the main edge dropout
            edge_index = torch.cat([edge_index, more_drop_edges]).unique()

        mask = ~index_to_mask(edge_index, graph.num_edges)

        graph = copy.copy(graph)
        graph.edge_index = graph.edge_index[:, mask]
        graph.edge_type = graph.edge_type[mask]

        return graph

    def execute(self, graph, query, symbolic_traversal):
        """Execute queries on the graph.
        symbolic_traversal is needed at training time for dropout
        and can be turned off for inference
        """
        self.symbolic_traversal = symbolic_traversal
        if self.training:
            assert self.symbolic_traversal is True, "symbolic_traversal is needed at train time for dropout"

        # we use stacks to execute postfix notations
        # check out this tutorial if you are not familiar with the algorithm
        # https://www.andrew.cmu.edu/course/15-121/lectures/Stacks%20and%20Queues/Stacks%20and%20Queues.html
        batch_size = len(query)
        # we execute a neural model and a symbolic model at the same time
        # the symbolic model is used for traversal dropout at training time
        # the elements in a stack are fuzzy sets
        self.stack = Stack(batch_size, self.stack_size, graph.num_nodes, device=query.device)
        self.var = Stack(batch_size, query.shape[1], graph.num_nodes, device=query.device)

        if self.symbolic_traversal:
            self.symbolic_stack = Stack(batch_size, self.stack_size, graph.num_nodes, device=query.device)
            self.symbolic_var = Stack(batch_size, query.shape[1], graph.num_nodes, device=query.device)
        
        # instruction pointer
        self.IP = torch.zeros(batch_size, dtype=torch.long, device=query.device)
        all_sample = torch.ones(batch_size, dtype=torch.bool, device=query.device)
        op = query[all_sample, self.IP]

        while not op.is_stop().all():
            is_operand = op.is_operand()
            is_intersection = op.is_intersection()
            is_union = op.is_union()
            is_negation = op.is_negation()
            is_projection = op.is_projection()
            if is_operand.any():
                h_index = op[is_operand].get_operand()
                self.apply_operand(is_operand, h_index, graph.num_nodes)
            if is_intersection.any():
                self.apply_intersection(is_intersection)
            if is_union.any():
                self.apply_union(is_union)
            if is_negation.any():
                self.apply_negation(is_negation)
            # only execute projection when there are no other operations
            # since projection is the most expensive and we want to maximize the parallelism
            if not (is_operand | is_negation | is_intersection | is_union).any() and is_projection.any():
                r_index = op[is_projection].get_operand()
                self.apply_projection(is_projection, graph, r_index)
            op = query[all_sample, self.IP]

        if (self.stack.SP > 1).any():
            raise ValueError("More operands than expected")

    def forward(self, graph, query, symbolic_traversal=True):
        self.execute(graph, query, symbolic_traversal)

        # convert probability to logit for compatibility reasons
        t_prob = self.stack.pop()
        t_logit = ((t_prob + 1e-10) / (1 - t_prob + 1e-10)).log()
        return t_logit


    def apply_operand(self, mask, h_index, num_node):
        h_prob = F.one_hot(h_index, num_node).float()
        self.stack.push(mask, h_prob)
        self.var.push(mask, h_prob)
        if self.symbolic_traversal:
            self.symbolic_stack.push(mask, h_prob)
            self.symbolic_var.push(mask, h_prob)
        self.IP[mask] += 1

    def apply_intersection(self, mask):
        y_prob = self.stack.pop(mask)
        x_prob = self.stack.pop(mask) 
        z_prob = self.conjunction(x_prob, y_prob)
        self.stack.push(mask, z_prob)
        self.var.push(mask, z_prob)
        if self.symbolic_traversal:
            sym_y_prob = self.symbolic_stack.pop(mask)
            sym_x_prob = self.symbolic_stack.pop(mask)
            sym_z_prob = self.conjunction(sym_x_prob, sym_y_prob)
            self.symbolic_stack.push(mask, sym_z_prob)
            self.symbolic_var.push(mask, sym_z_prob)
        self.IP[mask] += 1

    def apply_union(self, mask):
        y_prob = self.stack.pop(mask)
        x_prob = self.stack.pop(mask) 
        z_prob = self.disjunction(x_prob, y_prob)
        self.stack.push(mask, z_prob)
        self.var.push(mask, z_prob)
        if self.symbolic_traversal:
            sym_y_prob = self.symbolic_stack.pop(mask)
            sym_x_prob = self.symbolic_stack.pop(mask)
            sym_z_prob = self.disjunction(sym_x_prob, sym_y_prob)
            self.symbolic_stack.push(mask, sym_z_prob)
            self.symbolic_var.push(mask, sym_z_prob)
        self.IP[mask] += 1

    def apply_negation(self, mask):
        x_prob = self.stack.pop(mask)
        y_prob = self.negation(x_prob)
        self.stack.push(mask, y_prob)
        self.var.push(mask, y_prob)
        if self.symbolic_traversal:
            sym_x_prob = self.symbolic_stack.pop(mask)
            sym_y_prob = self.negation(sym_x_prob)
            self.symbolic_stack.push(mask, sym_y_prob)
            self.symbolic_var.push(mask, sym_y_prob)
        self.IP[mask] += 1

    def apply_projection(self, mask, graph, r_index):
        h_prob = self.stack.pop(mask)
        if self.training:
            sym_h_prob = self.symbolic_stack.pop(mask)
            # apply traversal dropout based on the output of the symbolic model
            graph = self.traversal_dropout(graph, sym_h_prob, r_index)
            # also changing the relation graph because of the changed main graph
            graph = build_relation_graph(graph)
        else:
            if self.symbolic_traversal:
                sym_h_prob = self.symbolic_stack.pop(mask)

        # detach the variable to stabilize training
        h_prob = h_prob.detach()
        t_prob = self.model(graph, h_prob, r_index)
        self.stack.push(mask, t_prob)
        self.var.push(mask, t_prob)

        if self.symbolic_traversal:
            sym_t_prob = self.symbolic_model(graph, sym_h_prob, r_index)
            self.symbolic_stack.push(mask, sym_t_prob)
            self.symbolic_var.push(mask, sym_t_prob)
            
        self.IP[mask] += 1

    def conjunction(self, x, y):
        if self.logic == "godel":
            return torch.min(x, y)
        elif self.logic == "product":
            return x * y
        elif self.logic == "lukasiewicz":
            return (x + y - 1).clamp(min=0)
        else:
            raise ValueError("Unknown fuzzy logic `%s`" % self.logic)

    def disjunction(self, x, y):
        if self.logic == "godel":
            return torch.max(x, y)
        elif self.logic == "product":
            return x + y - x * y
        elif self.logic == "lukasiewicz":
            return (x + y).clamp(max=1)
        else:
            raise ValueError("Unknown fuzzy logic `%s`" % self.logic)

    def negation(self, x):
        return 1 - x


class RelationProjection(nn.Module):
    """Wrap a GNN model for relation projection."""

    def __init__(self, model, threshold=0.0):
        super(RelationProjection, self).__init__()
        self.model = model
        self.threshold = threshold

    def forward(self, graph, h_prob, r_index):

        bs = r_index.shape[0]

        # GNN model: get relation representations conditioned on the query r_index
        rel_reprs = self.model.relation_model(graph.relation_graph, query=r_index) # (bs, num_rel, dim) 
        query = rel_reprs[torch.arange(bs, device=r_index.device), r_index]  # (bs, dim)

        # initialize the input with the fuzzy set and query relation
        input = torch.einsum("bn, bd -> bnd", h_prob, query)

        # GNNs trained on link prediction exhibit the multi-source propagation issue (see the paper)
        # We can partly alleviate it by thresholding intermediate scores
        if self.threshold > 0.0:
            temp_prob = h_prob.clone()
            # if self.threshold > 0.0:
            temp_prob[temp_prob <= self.threshold] = 0.0
            input = torch.einsum("bn, bd -> bnd", temp_prob, query)
        
        # GNN model: run the entity-level reasoner to get a scalar distribution over nodes
        output = self.model.entity_model(graph, input, rel_reprs, query)
        # Turn into probs
        t_prob = F.sigmoid(output)

        return t_prob



class SymbolicTraversal(nn.Module):
    """Symbolic traversal algorithm."""

    def forward(self, graph, h_prob, r_index):
        batch_size = len(h_prob)

        # For each query relation in the batch of r_index, we need to extract subgraphs induced by those edge types
        # OG torchdrug uses perfect hashing for the matching process, here we'll use vmap over the batch dim
        # Still not the most efficient method though, suggestions are welcome
        mask = torch.vmap(lambda t1, t2: t1 == t2 )(graph.edge_type.repeat(batch_size,1), r_index.unsqueeze(1))
        # Creating one big graph from all the subgraphs for one spmm_max function call
        graph = Batch.from_data_list([
            Data(edge_index=graph.edge_index[:, mask[i]], num_nodes=graph.num_nodes, device=graph.edge_index.device) for i in range(batch_size)
        ])

        t_prob = spmm_max(graph.edge_index.flip(0), torch.ones(graph.num_edges, device=h_prob.device), graph.num_nodes, graph.num_nodes, h_prob.view(-1, 1)).clamp(min=0)

        return t_prob.view_as(h_prob)
