import csv
import copy
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import distributed as dist

from torch_scatter import scatter_add, scatter_mean, scatter_max

from ultra import variadic, datasets_query


class Query(torch.Tensor):
    """Tensor storage of logical queries in postfix notations."""

    projection = 1 << 58
    intersection = 1 << 59
    union = 1 << 60
    negation = 1 << 61
    stop = 1 << 62
    operation = projection | intersection | union | negation | stop

    stack_size = 2

    def __new__(cls, data, device=None):
        query = torch.as_tensor(data, dtype=torch.long, device=device)
        query = torch.Tensor._make_subclass(cls, query)
        return query

    @classmethod
    def from_nested(cls, nested, binary_op=True):
        """Construct a logical query from nested tuples (BetaE format)."""
        if not binary_op:
            raise ValueError("The current implementation doesn't support nary operations")
        query = cls.nested_to_postfix(nested, binary_op=binary_op)
        query.append(cls.stop)
        return cls(query)

    @classmethod
    def nested_to_postfix(cls, nested, binary_op=True):
        """Recursively convert nested tuples into a postfix notation."""
        query = []

        if len(nested) == 2 and isinstance(nested[-1][-1], int):
            var, unary_ops = nested
            if isinstance(var, tuple):
                query += cls.nested_to_postfix(var, binary_op=binary_op)
            else:
                query.append(var)
            for op in unary_ops:
                if op == -2:
                    query.append(cls.negation)
                else:
                    query.append(cls.projection | op)
        else:
            if len(nested[-1]) > 1:
                vars, nary_op = nested, cls.intersection
            else:
                vars, nary_op = nested[:-1], cls.union
            num_args = 2 if binary_op else len(vars)
            op = nary_op | num_args
            for i, var in enumerate(vars):
                query += cls.nested_to_postfix(var)
                if i + 1 >= num_args:
                    query.append(op)

        return query

    def to_readable(self):
        """Convert this logical query to a human readable string."""
        if self.ndim > 1:
            raise ValueError("readable() can only be called for a single query")

        num_variable = 0
        stack = []
        lines = []
        for op in self:
            if op.is_operand():
                entity = op.get_operand().item()
                stack.append(str(entity))
            else:
                var = chr(ord("A") + num_variable)
                if op.is_projection():
                    relation = op.get_operand().item()
                    line = "%s <- projection_%d(%s)" % (var, relation, stack.pop())
                elif op.is_intersection():
                    num_args = op.get_operand()
                    args = stack[-num_args:]
                    stack = stack[:-num_args]
                    line = "%s <- intersection(%s)" % (var, ", ".join(args))
                elif op.is_union():
                    num_args = op.get_operand().item()
                    args = stack[-num_args:]
                    stack = stack[:-num_args]
                    line = "%s <- union(%s, %s)" % (var, ", ".join(args))
                elif op.is_negation():
                    line = "%s <- negation(%s)" % (var, stack.pop())
                elif op.is_stop():
                    break
                else:
                    raise ValueError("Unknown operator `%d`" % op)
                lines.append(line)
                stack.append(var)
                num_variable += 1

        if len(stack) > 1:
            raise ValueError("Invalid query. More operands than expected")
        line = "\n".join(lines)
        return line

    def computation_graph(self):
        """Get the computation graph of logical queries. Used for visualization."""
        query = self.view(-1, self.shape[-1])
        stack = Stack(len(query), self.stack_size, dtype=torch.long, device=query.device)
        # pointer to the next operator that consumes the output of this operator
        pointer = -torch.ones(query.shape, dtype=torch.long, device=query.device)
        # depth of each operator in the computation graph
        depth = -torch.ones(query.shape, dtype=torch.long, device=query.device)
        # width of the substree covered by each operator
        width = -torch.ones(query.shape, dtype=torch.long, device=query.device)

        for i, op in enumerate(query.t()):
            is_operand = op.is_operand()
            is_unary = op.is_projection() | op.is_negation()
            is_binary = op.is_intersection() | op.is_union()
            is_stop = op.is_stop()
            if is_operand.any():
                stack.push(is_operand, i)
                depth[is_operand, i] = 0
                width[is_operand, i] = 1
            if is_unary.any():
                prev = stack.pop(is_unary)
                pointer[is_unary, prev] = i
                depth[is_unary, i] = depth[is_unary, prev] + 1
                width[is_unary, i] = width[is_unary, prev]
                stack.push(is_unary, i)
            if is_binary.any():
                prev_y = stack.pop(is_binary)
                prev_x = stack.pop(is_binary)
                pointer[is_binary, prev_y] = i
                pointer[is_binary, prev_x] = i
                depth[is_binary, i] = torch.max(depth[is_binary, prev_x], depth[is_binary, prev_y]) + 1
                width[is_binary, i] = width[is_binary, prev_x] + width[is_binary, prev_y]
                stack.push(is_binary, i)
            if is_stop.all():
                break

        # each operator covers leaf nodes [left, right)
        left = torch.where(depth > 0, 0, -1)
        right = torch.where(depth > 0, width.max(), -1)
        # backtrack to update left and right
        for i in reversed(range(query.shape[-1])):
            has_pointer = pointer[:, i] != -1
            ptr = pointer[has_pointer, i]
            depth[has_pointer, i] = depth[has_pointer, ptr] - 1
            left[has_pointer, i] = left[has_pointer, ptr] + width[has_pointer, ptr] - width[has_pointer, i]
            right[has_pointer, i] = left[has_pointer, i] + width[has_pointer, i]
            width[has_pointer, ptr] -= width[has_pointer, i]

        pointer = pointer.view_as(self)
        depth = depth.view_as(self)
        left = left.view_as(self)
        right = right.view_as(self)
        return pointer, depth, left, right

    def is_operation(self):
        return (self & self.operation > 0)

    def is_operand(self):
        return ~(self & self.operation > 0)

    def is_projection(self):
        return self & self.projection > 0

    def is_intersection(self):
        return self & self.intersection > 0

    def is_union(self):
        return self & self.union > 0

    def is_negation(self):
        return self & self.negation > 0

    def is_stop(self):
        return self & self.stop > 0

    def get_operation(self):
        return self & self.operation

    def get_operand(self):
        return self & ~self.operation

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class Stack(object):
    """
    Batch of stacks implemented in PyTorch.

    Parameters:
        batch_size (int): batch size
        stack_size (int): max stack size
        shape (tuple of int, optional): shape of each element in the stack
        dtype (torch.dtype): dtype
        device (torch.device): device
    """

    def __init__(self, batch_size, stack_size, *shape, dtype=None, device=None):
        self.stack = torch.zeros(batch_size, stack_size, *shape, dtype=dtype, device=device)
        self.SP = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.batch_size = batch_size
        self.stack_size = stack_size

    def push(self, mask, value):
        if (self.SP[mask] >= self.stack_size).any():
            raise ValueError("Stack overflow")
        self.stack[mask, self.SP[mask]] = value
        self.SP[mask] += 1

    def pop(self, mask=None):
        if (self.SP[mask] < 1).any():
            raise ValueError("Stack underflow")
        if mask is None:
            mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.stack.device)
        self.SP[mask] -= 1
        return self.stack[mask, self.SP[mask]]

    def top(self, mask=None):
        if (self.SP < 1).any():
            raise ValueError("Stack is empty")
        if mask is None:
            mask = torch.ones(self.batch_size, dtype=torch.bool, device=self.stack.device)
        return self.stack[mask, self.SP[mask] - 1]


def gather_results(pred, target, rank, world_size, device):
    # for multi-gpu setups: join results together 
    # for single-gpu setups: doesn't do anything special
    ranking, num_pred = pred
    type, answer_ranking, num_easy, num_hard = target

    all_size_r = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_ar = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_p = torch.zeros(world_size, dtype=torch.long, device=device)
    all_size_r[rank] = len(ranking)
    all_size_ar[rank] = len(answer_ranking)
    all_size_p[rank] = len(num_pred)
    if world_size > 1:
        dist.all_reduce(all_size_r, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_ar, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_size_p, op=dist.ReduceOp.SUM)

    # obtaining all ranks 
    cum_size_r = all_size_r.cumsum(0)
    cum_size_ar = all_size_ar.cumsum(0)
    cum_size_p = all_size_p.cumsum(0)

    all_ranking = torch.zeros(all_size_r.sum(), dtype=torch.long, device=device)
    all_num_pred = torch.zeros(all_size_p.sum(), dtype=torch.long, device=device)
    all_types = torch.zeros(all_size_p.sum(), dtype=torch.long, device=device)
    all_answer_ranking = torch.zeros(all_size_ar.sum(), dtype=torch.long, device=device)
    all_num_easy = torch.zeros(all_size_p.sum(), dtype=torch.long, device=device)
    all_num_hard = torch.zeros(all_size_p.sum(), dtype=torch.long, device=device)

    all_ranking[cum_size_r[rank] - all_size_r[rank]: cum_size_r[rank]] = ranking
    all_num_pred[cum_size_p[rank] - all_size_p[rank]: cum_size_p[rank]] = num_pred
    all_types[cum_size_p[rank] - all_size_p[rank]: cum_size_p[rank]] = type
    all_answer_ranking[cum_size_ar[rank] - all_size_ar[rank]: cum_size_ar[rank]] = answer_ranking
    all_num_easy[cum_size_p[rank] - all_size_p[rank]: cum_size_p[rank]] = num_easy
    all_num_hard[cum_size_p[rank] - all_size_p[rank]: cum_size_p[rank]] = num_hard

    if world_size > 1:
        dist.all_reduce(all_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_pred, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_types, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_answer_ranking, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_easy, op=dist.ReduceOp.SUM)
        dist.all_reduce(all_num_hard, op=dist.ReduceOp.SUM)
    
    return (all_ranking.cpu(), all_num_pred.cpu()), (all_types.cpu(), all_answer_ranking.cpu(), all_num_easy.cpu(), all_num_hard.cpu())

def batch_evaluate(pred, target, limit_nodes=None):
    type, easy_answer, hard_answer = target

    num_easy = easy_answer.sum(dim=-1)
    num_hard = hard_answer.sum(dim=-1)
    num_answer = num_easy + num_hard
    # answer2query = functional._size_to_index(num_answer)
    answer2query = torch.repeat_interleave(num_answer)

    num_entity = pred.shape[-1]

    # in inductive (e) fb_ datasets, the number of nodes in the graph structure might exceed
    # the actual number of nodes in the graph, so we'll mask unused nodes
    if limit_nodes is not None:
            # print(f"Keeping only {len(limit_nodes)} nodes out of {num_entity}")
            keep_mask = torch.zeros(num_entity, dtype=torch.bool, device=limit_nodes.device)
            keep_mask[limit_nodes] = 1
            #keep_mask = F.one_hot(limit_nodes, num_entity)
            pred[:, ~keep_mask] = float('-inf')
    
    order = pred.argsort(dim=-1, descending=True)

    range = torch.arange(num_entity, device=pred.device)
    ranking = scatter_add(range.expand_as(order), order, dim=-1)

    easy_ranking = ranking[easy_answer]
    hard_ranking = ranking[hard_answer]
    # unfiltered rankings of all answers
    answer_ranking = variadic._extend(easy_ranking, num_easy, hard_ranking, num_hard)[0]
    order_among_answer = variadic.variadic_sort(answer_ranking, num_answer)[1]
    order_among_answer = order_among_answer + (num_answer.cumsum(0) - num_answer)[answer2query]
    ranking_among_answer = scatter_add(variadic.variadic_arange(num_answer), order_among_answer)

    # filtered rankings of all answers
    ranking = answer_ranking - ranking_among_answer + 1
    ends = num_answer.cumsum(0)
    starts = ends - num_hard
    hard_mask = variadic.multi_slice_mask(starts, ends, ends[-1])
    # filtered rankings of hard answers
    ranking = ranking[hard_mask]

    return ranking, answer_ranking

def evaluate(pred, target, metrics, id2type):
    ranking, num_pred = pred
    type, answer_ranking, num_easy, num_hard = target

    metric = {}
    for _metric in metrics:
        if _metric == "mrr":
            answer_score = 1 / ranking.float()
            query_score = variadic.variadic_mean(answer_score, num_hard)
            type_score = scatter_mean(query_score, type, dim_size=len(id2type))
        elif _metric.startswith("hits@"):
            threshold = int(_metric[5:])
            answer_score = (ranking <= threshold).float()
            query_score = variadic.variadic_mean(answer_score, num_hard)
            type_score = scatter_mean(query_score, type, dim_size=len(id2type))
        elif _metric == "mape":
            query_score = (num_pred - num_easy - num_hard).abs() / (num_easy + num_hard).float()
            type_score = scatter_mean(query_score, type, dim_size=len(id2type))
        elif _metric == "spearmanr":
            type_score = []
            for i in range(len(id2type)):
                mask = type == i
                score = spearmanr(num_pred[mask], num_easy[mask] + num_hard[mask])
                type_score.append(score)
            type_score = torch.stack(type_score)
        elif _metric == "auroc":
            ends = (num_easy + num_hard).cumsum(0)
            starts = ends - num_hard
            target = variadic.multi_slice_mask(starts, ends, len(answer_ranking)).float()
            answer_score = variadic_area_under_roc(answer_ranking, target, num_easy + num_hard)
            mask = (num_easy > 0) & (num_hard > 0)
            query_score = answer_score[mask]
            type_score = scatter_mean(query_score, type[mask], dim_size=len(id2type))
        else:
            raise ValueError("Unknown metric `%s`" % _metric)

        score = type_score.mean()
        is_neg = torch.tensor(["n" in t for t in id2type], device=ranking.device)
        is_epfo = ~is_neg
        name = _metric
        for i, query_type in enumerate(id2type):
            metric["[%s] %s" % (query_type, name)] = type_score[i].item()
        if is_epfo.any():
            epfo_score = variadic.masked_mean(type_score, is_epfo)
            metric["[EPFO] %s" % name] = epfo_score.item()
        if is_neg.any():
            neg_score = variadic.masked_mean(type_score, is_neg)
            metric["[negation] %s" % name] = neg_score.item()
        metric[name] = score.item()

    return metric

def variadic_area_under_roc(pred, target, size):
    """
    Area under receiver operating characteristic curve (ROC) for sets with variadic sizes.

    Suppose there are :math:`N` sets, and the sizes of all sets are summed to :math:`B`.

    Parameters:
        pred (Tensor): prediction of shape :math:`(B,)`
        target (Tensor): target of shape :math:`(B,)`.
        size (Tensor): size of sets of shape :math:`(N,)`
    """

    index2graph = torch.repeat_interleave(size)
    _, order = variadic.variadic_sort(pred, size, descending=True)
    cum_size = (size.cumsum(0) - size)[index2graph]
    target = target[order + cum_size]
    total_hit = variadic.variadic_sum(target, size)
    total_hit = total_hit.cumsum(0) - total_hit
    hit = target.cumsum(0) - total_hit[index2graph]
    hit = torch.where(target == 0, hit, torch.zeros_like(hit))
    all = variadic.variadic_sum((target == 0).float(), size) * \
            variadic.variadic_sum((target == 1).float(), size)
    auroc = variadic.variadic_sum(hit, size) / (all + 1e-10)
    return auroc

def spearmanr(pred, target):
    """
    Spearman correlation between prediction and target.

    Parameters:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`
    """

    def get_ranking(input):
        input_set, input_inverse = input.unique(return_inverse=True)
        order = input_inverse.argsort()
        ranking = torch.zeros(len(input_inverse), device=input.device)
        ranking[order] = torch.arange(1, len(input) + 1, dtype=torch.float, device=input.device)

        # for elements that have the same value, replace their rankings with the mean of their rankings
        mean_ranking = scatter_mean(ranking, input_inverse, dim=0, dim_size=len(input_set))
        ranking = mean_ranking[input_inverse]
        return ranking

    pred = get_ranking(pred)
    target = get_ranking(target)
    covariance = (pred * target).mean() - pred.mean() * target.mean()
    pred_std = pred.std(unbiased=False)
    target_std = target.std(unbiased=False)
    spearmanr = covariance / (pred_std * target_std + 1e-10)
    return spearmanr


def spmm_max(index: Tensor, value: Tensor, m: int, n: int,
         matrix: Tensor) -> Tensor:
    """
    The same spmm kernel from torch_sparse 
    https://github.com/rusty1s/pytorch_sparse/blob/master/torch_sparse/spmm.py#L29

    with the only change that instead of scatter_add aggregation 
    we keep scatter_max

    Matrix product of sparse matrix with dense matrix.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix, either of
            floating-point or integer type. Does not work for boolean and
            complex number data types.
        m (int): The first dimension of sparse matrix.
        n (int): The second dimension of sparse matrix.
        matrix (:class:`Tensor`): The dense matrix of same type as
            :obj:`value`.

    :rtype: :class:`Tensor`
    """

    assert n == matrix.size(-2)

    row, col = index[0], index[1]
    matrix = matrix if matrix.dim() > 1 else matrix.unsqueeze(-1)

    out = matrix.index_select(-2, col)
    out = out * value.unsqueeze(-1)
    out = scatter_max(out, row, dim=-2, dim_size=m)[0]

    return out

def build_query_dataset(cfg):
    data_config = copy.deepcopy(cfg.dataset)
    cls = data_config.pop("class")

    ds_cls = getattr(datasets_query, cls)
    dataset = ds_cls(**data_config)

    return dataset

def cat(objs, *args, **kwargs):
    """
    Concatenate a list of nested containers with the same structure.
    """
    obj = objs[0]
    if isinstance(obj, torch.Tensor):
        return torch.cat(objs, *args, **kwargs)
    elif isinstance(obj, dict):
        return {k: cat([x[k] for x in objs], *args, **kwargs) for k in obj}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(cat(xs, *args, **kwargs) for xs in zip(*objs))

    raise TypeError("Can't perform concatenation over object type `%s`" % type(obj))

def print_metrics(metrics, logger, roundto=4):
    order = sorted(list(metrics.keys()))
    for key in order:
        logger.warning(f"{key}: {round(metrics[key], roundto)}")

def print_metrics_to_file(metrics, results_file, roundto=4):
    # round up all values in the metrics dict
    metrics = {k: round(v,roundto) if type(v).__name__ != "str" else v for k,v in metrics.items()}
    with open(results_file, "a", newline='') as csv_file:
        fieldnames = sorted(list(metrics.keys())) 
        fieldnames.remove("dataset")
        fieldnames = ['dataset']+fieldnames
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=',')
        if csv_file.tell() == 0:
            writer.writeheader()
        writer.writerow(metrics)

def cuda(obj, *args, **kwargs):
    """
    Transfer any nested container of tensors to CUDA.
    """
    if hasattr(obj, "cuda"):
        return obj.cuda(*args, **kwargs)
    elif isinstance(obj, (str, bytes)):
        return obj
    elif isinstance(obj, dict):
        return type(obj)({k: cuda(v, *args, **kwargs) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return type(obj)(cuda(x, *args, **kwargs) for x in obj)

    raise TypeError("Can't transfer object type `%s`" % type(obj))