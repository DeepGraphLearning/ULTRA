import os
import pickle
from collections import defaultdict
from tqdm import tqdm

import torch
import numpy as np
from torch.nn import functional as F
from torch.utils import data as torch_data
from functools import partial

from torch_scatter import scatter_add
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip

from ultra.query_utils import Query
from ultra.tasks import build_relation_graph
from ultra.base_nbfnet import index_to_mask


class LogicalQueryDataset(InMemoryDataset):
    """Logical query dataset."""

    struct2type = {
        ("e", ("r",)): "1p",
        ("e", ("r", "r")): "2p",
        ("e", ("r", "r", "r")): "3p",
        (("e", ("r",)), ("e", ("r",))): "2i",
        (("e", ("r",)), ("e", ("r",)), ("e", ("r",))): "3i",
        ((("e", ("r",)), ("e", ("r",))), ("r",)): "ip",
        (("e", ("r", "r")), ("e", ("r",))): "pi",
        (("e", ("r",)), ("e", ("r", "n"))): "2in",
        (("e", ("r",)), ("e", ("r",)), ("e", ("r", "n"))): "3in",
        ((("e", ("r",)), ("e", ("r", "n"))), ("r",)): "inp",
        (("e", ("r", "r")), ("e", ("r", "n"))): "pin",
        (("e", ("r", "r", "n")), ("e", ("r",))): "pni",
        (("e", ("r",)), ("e", ("r",)), ("u",)): "2u-DNF",
        ((("e", ("r",)), ("e", ("r",)), ("u",)), ("r",)): "up-DNF",
        ((("e", ("r", "n")), ("e", ("r", "n"))), ("n",)): "2u-DM",
        ((("e", ("r", "n")), ("e", ("r", "n"))), ("n", "r")): "up-DM",
    }

    def __init__(self, root, transform=None, pre_transform=build_relation_graph, 
                 query_types=None, union_type="DNF", train_patterns = None, **kwargs):

        self.query_types = query_types
        self.union_type = union_type
        self.train_patterns = train_patterns
        super().__init__(root, transform, pre_transform)
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["train.txt", "valid.txt", "test.txt"]
    
    def download(self):
        download_path = download_url(self.url, self.root)
        extract_zip(download_path, self.root)
        # os.unlink(download_path)

    def set_query_types(self):
        query_types = self.query_types or self.struct2type.values()
        new_query_types = []
        for query_type in query_types:
            if "u" in query_type:
                if "-" not in query_type:
                    query_type = "%s-%s" % (query_type, self.union_type)
                elif query_type[query_type.find("-") + 1:] != self.union_type:
                    continue
            new_query_types.append(query_type)
        self.id2type = sorted(new_query_types)
        self.type2id = {t: i for i, t in enumerate(self.id2type)}

    def process(self):
        """
        Load the dataset from pickle dumps (BetaE format).

        Parameters:
            path (str): path to pickle dumps
            query_types (list of str, optional): query types to load.
                By default, load all query types.
            union_type (str, optional): which union type to use, ``DNF`` or ``DM``
            verbose (int, optional): output verbose level
        """
        self.set_query_types()
        path = self.raw_dir

        with open(os.path.join(path, "id2ent.pkl"), "rb") as fin:
            entity_vocab = pickle.load(fin)
        with open(os.path.join(path, "id2rel.pkl"), "rb") as fin:
            relation_vocab = pickle.load(fin)
        triplets = []
        num_samples = []
        for split in ["train", "valid", "test"]:
            triplet_file = os.path.join(path, "%s.txt" % split)
            with open(triplet_file) as fin:
                num_sample = 0
                for line in fin:
                    h, r, t = [int(x) for x in line.split()]
                    triplets.append((h, t, r))
                    num_sample += 1
                num_samples.append(num_sample)
        
        train_edges = torch.tensor([[t[0], t[1]] for t in triplets[:num_samples[0]]], dtype=torch.long).t()
        train_edge_types = torch.tensor([t[2] for t in triplets[:num_samples[0]]], dtype=torch.long)

        # The 'inverse_rel_plus_one' property is needed for traversal dropout to determine the way of deriving inverse edges
        # In BetaE datasets, inv_rel = direct_rel + 1, but in inducitve datasets it is inv_rel = direct_rel + num_relations
        self.train_graph = Data(edge_index=train_edges, edge_type=train_edge_types, 
                               num_nodes=len(entity_vocab), num_relations=len(relation_vocab), inverse_rel_plus_one=True)
        self.valid_graph = self.train_graph
        self.test_graph = self.train_graph
        
        if self.pre_transform is not None:
            self.train_graph = self.pre_transform(self.train_graph)

        # loading queries
        queries = []
        types = []
        easy_answers = []
        hard_answers = []
        num_samples = []
        max_query_length = 0

        for split in ["train", "valid", "test"]:
            
            pbar = tqdm(desc="Loading %s-*.pkl" % split, total=3)
            with open(os.path.join(path, "%s-queries.pkl" % split), "rb") as fin:
                struct2queries = pickle.load(fin)
            pbar.update(1)
            type2queries = {self.struct2type[k]: v for k, v in struct2queries.items()}
            type2queries = {k: v for k, v in type2queries.items() if k in self.type2id}
            if split == "train":
                with open(os.path.join(path, "%s-answers.pkl" % split), "rb") as fin:
                    query2easy_answers = pickle.load(fin)
                query2hard_answers = defaultdict(set)
                pbar.update(2)
            else:
                with open(os.path.join(path, "%s-easy-answers.pkl" % split), "rb") as fin:
                    query2easy_answers = pickle.load(fin)
                pbar.update(1)
                with open(os.path.join(path, "%s-hard-answers.pkl" % split), "rb") as fin:
                    query2hard_answers = pickle.load(fin)
                pbar.update(1)

            num_sample = sum([len(q) for t, q in type2queries.items()])
            pbar = tqdm(desc="Processing %s queries" % split, total=num_sample)
            for type in type2queries:
                struct_queries = sorted(type2queries[type])
                for query in struct_queries:
                    easy_answers.append(query2easy_answers[query])
                    hard_answers.append(query2hard_answers[query])
                    query = Query.from_nested(query)
                    queries.append(query)
                    max_query_length = max(max_query_length, len(query))
                    types.append(self.type2id[type])
                    pbar.update(1)
            num_samples.append(num_sample)

        self.queries = queries
        self.types = types
        self.easy_answers = easy_answers
        self.hard_answers = hard_answers
        self.num_samples = num_samples
        self.max_query_length = max_query_length

    def __getitem__(self, index):
        query = self.queries[index]
        easy_answer = torch.tensor(list(self.easy_answers[index]), dtype=torch.long)
        hard_answer = torch.tensor(list(self.hard_answers[index]), dtype=torch.long)
        return {
            "query": F.pad(query, (0, self.max_query_length - len(query)), value=query.stop),
            "type": self.types[index],
            "easy_answer": index_to_mask(easy_answer, self.train_graph.num_nodes),
            "hard_answer": index_to_mask(hard_answer, self.train_graph.num_nodes),
        }

    def __len__(self):
        return len(self.queries)

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, offset + num_sample))
            splits.append(split)
            offset += num_sample
        return splits
    
    def __repr__(self):
        return "%s()" % (self.name)
    
    @property
    def num_relations(self):
        return int(self.train_graph.num_relations)

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name)  # +raw

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name)  # + processed

    @property
    def processed_file_names(self):
        return "data.pt"


class FB15kLogicalQuery(LogicalQueryDataset):

    name = "FB15k-betae"
    url = "http://snap.stanford.edu/betae/KG_data.zip"
    md5 = "d54f92e2e6a64d7f525b8fe366ab3f50"


class FB15k237LogicalQuery(LogicalQueryDataset):

    name = "FB15k-237-betae"
    url = "http://snap.stanford.edu/betae/KG_data.zip"
    md5 = "d54f92e2e6a64d7f525b8fe366ab3f50"


class NELL995LogicalQuery(LogicalQueryDataset):

    name = "NELL-betae"
    url = "http://snap.stanford.edu/betae/KG_data.zip"
    md5 = "d54f92e2e6a64d7f525b8fe366ab3f50"


class InductiveFB15k237Query(LogicalQueryDataset):

    url = "https://zenodo.org/record/7306046/files/%s.zip"

    md5 = {
        550: "e78bb9a7de9bd55813bb17f57941303c",
        300: "4db5c172acf83f676c9cf6589e033d7e",
        217: "9fde4563c619dc4d2b81af200cf7bc6b",
        175: "29ee1dbed7662740a2f001a0c6df8911",
        150: "61b545de8e5cdb04832f27842d8c0175",
        134: "cd8028c9674dc81f38cd17b03af43fe1",
        122: "272d2cc1e3f98f76d02daaf066f9d653",
        113: "e4ea60448e918c62779cfa757a096aa9",
        106: "6f9a1dcf22108074fb94a05b8377a173",
        "wikikg": "fa30b189436ab46a2ff083dd6a5e6e0b"
    }

    @property
    def raw_file_names(self):
        return [f"train_queries.pkl", "valid_queries.pkl", "test_queries.pkl"]

    def __init__(self, root, version, transform=None, pre_transform=build_relation_graph, query_types=None, union_type="DNF", 
                 train_patterns=('1p', '2p', '3p', '2i', '3i', '2in', '3in', 'inp', 'pni', 'pin'), **kwargs):
        self.version = version
        super().__init__(root, transform, pre_transform, query_types=query_types, union_type=union_type, train_patterns=train_patterns)

    def download(self):
        download_path = download_url(self.url % self.version, self.root)
        extract_zip(download_path, self.root)
        # os.unlink(download_path)

    def process(self):

        self.set_query_types()
        path = self.raw_dir

        # Space of entities 0 ... N is split into 3 sets
        # Train node IDs: 0 ... K
        # Val inference ids: K ... K+M
        # Test inference ids: K+M .... N
        try:
            train_triplets = self.load_file(os.path.join(path, "train_graph.txt"))
            val_inference = self.load_file(os.path.join(path, "val_inference.txt"))
            test_inference = self.load_file(os.path.join(path, "test_inference.txt"))
        except FileNotFoundError:
            print("Loading .pt files")
            train_triplets = self.load_pt(os.path.join(path, "train_graph.pt"))
            val_inference = self.load_pt(os.path.join(path, "val_inference.pt"))
            test_inference = self.load_pt(os.path.join(path, "test_inference.pt"))

        entity_vocab, relation_vocab, inv_ent_vocab, inv_rel_vocab, \
            tr_nodes, vl_nodes, ts_nodes = self.build_vocab(train_triplets, val_inference, test_inference)

        num_node = len(entity_vocab) if entity_vocab else None
        num_relation = len(relation_vocab) if relation_vocab else None

        # Training graph: only training triples
        self.train_graph = Data(edge_index=torch.LongTensor(train_triplets)[:, :2].t(), 
                                edge_type=torch.LongTensor(train_triplets)[:, 2],
                                num_nodes=len(tr_nodes), num_relations=num_relation)

        # Validation graph: training triples (0..K) + new validation inference triples (K+1...K+M)
        self.valid_graph = Data(edge_index=torch.LongTensor(train_triplets + val_inference)[:, :2].t(),
                                edge_type=torch.LongTensor(train_triplets + val_inference)[:, 2],
                                num_nodes=num_node, num_relations=num_relation,
                                restrict_nodes=torch.LongTensor(vl_nodes)  # need those for evaluation 
                                )

        # Test graph: training triples (0..K) + new test inference triples (K+M+1... N)
        self.test_graph = Data(edge_index=torch.LongTensor(train_triplets + test_inference)[:, :2].t(), 
                               edge_type=torch.LongTensor(train_triplets + test_inference)[:, 2],
                               num_nodes=num_node, num_relations=num_relation,
                               restrict_nodes=torch.LongTensor(ts_nodes),  # need those for evaluation
                               )

        if self.pre_transform:
            self.train_graph = self.pre_transform(self.train_graph)
            self.valid_graph = self.pre_transform(self.valid_graph)
            self.test_graph = self.pre_transform(self.test_graph)

        # Full graph (aux purposes)
        self.graph = Data(edge_index=torch.LongTensor(train_triplets + val_inference + test_inference)[:, :2].t(),
                          edge_type=torch.LongTensor(train_triplets + val_inference + test_inference)[:, 2],
                          num_nodes=num_node, num_relations=num_relation)
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab
        self.inv_entity_vocab = inv_ent_vocab
        self.inv_relation_vocab = inv_rel_vocab

        # Need those for evaluation
        self.valid_nodes = torch.LongTensor(vl_nodes)
        self.test_nodes = torch.LongTensor(ts_nodes)

        self.load_queries(path=path)

    def load_queries(self, path):

        queries = []
        type_ids = []
        easy_answers = []
        hard_answers = []
        num_samples = []
        num_entity_for_sample = []
        max_query_length = 0

        type2struct = {v: k for k, v in self.struct2type.items()}
        filtered_training_structs = tuple([type2struct[x] for x in self.train_patterns])
        for split in ["train", "valid", "test"]:
            with open(os.path.join(path, "%s_queries.pkl" % split), "rb") as fin:
                struct2queries = pickle.load(fin)
            if split == "train":
                query2hard_answers = defaultdict(lambda: defaultdict(set))
                with open(os.path.join(path, "%s_answers_hard.pkl" % split), "rb") as fin:
                    query2easy_answers = pickle.load(fin)
            else:
                with open(os.path.join(path, "%s_answers_easy.pkl" % split), "rb") as fin:
                    query2easy_answers = pickle.load(fin)
                with open(os.path.join(path, "%s_answers_hard.pkl" % split), "rb") as fin:
                    query2hard_answers = pickle.load(fin)
            num_sample = 0
            structs = sorted(struct2queries.keys(), key=lambda s: self.struct2type[s])
            structs = tqdm(structs, "Loading %s queries" % split)
            for struct in structs:
                query_type = self.struct2type[struct]
                if query_type not in self.type2id:
                    continue
                # filter complex patterns ip, pi, 2u, up from training queries - those will be eval only
                if split == "train" and struct not in filtered_training_structs:
                    print(f"Skipping {query_type} - this will be used in evaluation")
                    continue
                struct_queries = sorted(struct2queries[struct])
                for query in struct_queries:
                    # The dataset format is slightly different from BetaE's
                    easy_answers.append(query2easy_answers[struct][query])
                    hard_answers.append(query2hard_answers[struct][query])
                    query = Query.from_nested(query)
                    #query = self.to_postfix_notation(query)
                    max_query_length = max(max_query_length, len(query))
                    queries.append(query)
                    type_ids.append(self.type2id[query_type])
                num_sample += len(struct_queries)
            num_entity_for_sample += [getattr(self, "%s_graph" % split).num_nodes] * num_sample
            num_samples.append(num_sample)

        self.queries = queries
        self.types = type_ids
        self.easy_answers = easy_answers
        self.hard_answers = hard_answers
        self.num_samples = num_samples
        self.num_entity_for_sample = num_entity_for_sample
        self.max_query_length = max_query_length

    def load_file(self, path):
        triplets = []
        with open(path) as fin:
            for line in fin:
                h, r, t = [int(x) for x in line.split()]
                triplets.append((h, t, r))

        return triplets

    def load_pt(self, path):
        triplets = torch.load(path, map_location="cpu")
        return triplets[:, [0, 2, 1]].tolist()

    def build_vocab(self, train_triples, val_triples, test_triples):
        # datasets are already shipped with contiguous node IDs from 0 to N, so the total num ents is N+1
        all_triples = np.array(train_triples+val_triples+test_triples)
        train_nodes = np.unique(np.array(train_triples)[:, [0, 1]])
        val_nodes = np.unique(np.array(train_triples + val_triples)[:, [0, 1]])
        test_nodes = np.unique(np.array(train_triples + test_triples)[:, [0, 1]])
        num_entities = np.max(all_triples[:, [0, 1]]) + 1
        num_relations = np.max(all_triples[:, 2]) + 1

        ent_vocab = {i: i for i in range(num_entities)}
        rel_vocab = {i: i for i in range(num_relations)}
        inv_ent_vocab = {v:k for k,v in ent_vocab.items()}
        inv_rel_vocab = {v:k for k,v in rel_vocab.items()}

        return ent_vocab, rel_vocab, inv_ent_vocab, inv_rel_vocab, train_nodes, val_nodes, test_nodes

    def __getitem__(self, index):
        query = self.queries[index]
        easy_answer = torch.tensor(list(self.easy_answers[index]), dtype=torch.long)
        hard_answer = torch.tensor(list(self.hard_answers[index]), dtype=torch.long)
        # num_entity in the inductive setup is different for different splits, take it from the relevant graph
        num_entity = self.num_entity_for_sample[index]
        return {
            "query": F.pad(query, (0, self.max_query_length - len(query)), value=query.stop),
            "type": self.types[index],
            "easy_answer": index_to_mask(easy_answer, num_entity),
            "hard_answer": index_to_mask(hard_answer, num_entity),
        }

    @property
    def name(self):
        return f"{self.version}"
    
    def __repr__(self):
        return f"fb_{self.version}"
    

class WikiTopicsQuery(InductiveFB15k237Query):

    url = "https://reltrans.s3.us-east-2.amazonaws.com/WikiTopics_QE.zip"
    md5 = None

    @property
    def raw_file_names(self):
        return [f"train_queries.pkl", "valid_queries.pkl", "test_queries.pkl"]

    def __init__(self, root, version, transform=None, pre_transform=build_relation_graph, query_types=None, union_type="DNF",
                 train_patterns=('1p', '2p', '3p', '2i', '3i', '2in', '3in', 'inp', 'pni', 'pin'), **kwargs):
        #self.version = version
        super().__init__(root, version, transform, pre_transform, query_types=query_types, union_type=union_type, train_patterns=train_patterns)

    def download(self):
        download_path = download_url(self.url, self.root)
        extract_zip(download_path, self.root)
        # os.unlink(download_path)

    def process(self):

        self.set_query_types()
        # Download data if it's not there -> Add wt prefix?
        path = self.raw_dir

        # WikiTopics are standard inductive datasets: train/valid graph is separated from the test
        # Space of entities 0 ... N is split into 3 sets
        # Train node IDs: 0 ... K
        # Val inference ids: 0 ..., K
        # Test inference ids: O ..., M
        try:
            train_triplets = self.load_file(os.path.join(path, "train_graph.txt"))
            val_inference = self.load_file(os.path.join(path, "val_inference.txt"))
            test_inference = self.load_file(os.path.join(path, "test_inference.txt"))
        except FileNotFoundError:
            print("Loading .pt files")
            train_triplets = self.load_pt(os.path.join(path, "train_graph.pt"))
            val_inference = self.load_pt(os.path.join(path, "val_inference.pt"))
            test_inference = self.load_pt(os.path.join(path, "test_inference.pt"))

        train_entity_vocab, train_rel_vocab, test_ent_vocab, test_rel_vocab, \
            tr_nodes, vl_nodes, ts_nodes = self.build_vocab(train_triplets, val_inference, test_inference)

        # Training graph: only training triples
        self.train_graph = Data(edge_index=torch.LongTensor(train_triplets)[:, :2].t(), 
                                edge_type=torch.LongTensor(train_triplets)[:, 2],
                                num_nodes=len(tr_nodes), num_relations=len(train_rel_vocab))

        # Validation graph: the same as training
        self.valid_graph = Data(edge_index=torch.LongTensor(train_triplets)[:, :2].t(), 
                                edge_type=torch.LongTensor(train_triplets)[:, 2],
                                num_nodes=len(tr_nodes), num_relations=len(train_rel_vocab))
        
        # Test graph: a new graph with new entities/relations
        self.test_graph = Data(edge_index=torch.LongTensor(test_inference)[:, :2].t(), 
                                edge_type=torch.LongTensor(test_inference)[:, 2],
                                num_nodes=len(ts_nodes), num_relations=len(test_rel_vocab))
        if self.pre_transform:
            self.train_graph = self.pre_transform(self.train_graph)
            self.valid_graph = self.pre_transform(self.valid_graph)
            self.test_graph = self.pre_transform(self.test_graph)

        # dummy graph for compatibility purposes
        self.graph = self.test_graph

        # Full graph (aux purposes)
        # self.full_graph_valid = data.Graph(train_triplets + val_inference + test_inference,
        #                         num_node=num_node, num_relation=num_relation)
        self.train_entity_vocab, self.inv_train_ent_vocab = train_entity_vocab, {v:k for k,v in train_entity_vocab.items()}
        self.train_relation_vocab, self.inv_train_rel_vocab = train_rel_vocab, {v:k for k,v in train_rel_vocab.items()}
        self.test_entity_vocab, self.inv_test_ent_vocab = test_ent_vocab, {v:k for k,v in test_ent_vocab.items()}
        self.test_relation_vocab, self.inv_test_rel_vocab = test_rel_vocab, {v:k for k,v in test_rel_vocab.items()}

        # Need those for evaluation
        self.valid_nodes = torch.tensor(vl_nodes, dtype=torch.long)
        self.test_nodes = torch.tensor(ts_nodes, dtype=torch.long)

        self.load_queries(path)

    def build_vocab(self, train_triples, val_triples, test_triples):
        # In WikiTopics, validation graph is the same as train, but test is different
        train_triples, test_triples = np.array(train_triples), np.array(test_triples)
        train_nodes = np.unique(train_triples[:, [0, 1]])
        #val_nodes = np.unique(np.array(train_triples + val_triples)[:, [0, 1]])
        test_nodes = np.unique(test_triples[:, [0, 1]])

        num_train_entities = len(train_nodes)
        num_test_entities = len(test_nodes)
        num_train_relations = np.max(train_triples[:, 2]) + 1
        num_test_relations = np.max(test_triples[:, 2]) + 1

        train_ent_vocab = {i: i for i in range(num_train_entities)}
        train_rel_vocab = {i: i for i in range(num_train_relations)}
        test_ent_vocab = {i: i for i in range(num_test_entities)}
        test_rel_vocab = {i: i for i in range(num_test_relations)}

        return train_ent_vocab, train_rel_vocab, test_ent_vocab, test_rel_vocab, train_nodes, train_nodes, test_nodes

    def __getitem__(self, index):
        query = self.queries[index]
        easy_answer = torch.tensor(list(self.easy_answers[index]), dtype=torch.long)
        hard_answer = torch.tensor(list(self.hard_answers[index]), dtype=torch.long)
        # num_entity in the inductive setup is different for different splits, take it from the relevant graph
        num_entity = self.num_entity_for_sample[index]
        return {
            "query": F.pad(query, (0, self.max_query_length - len(query)), value=query.stop),
            "type": self.types[index],
            "easy_answer": index_to_mask(easy_answer, num_entity),
            "hard_answer": index_to_mask(hard_answer, num_entity),
        }
    
    @property
    def raw_dir(self):
        return os.path.join(self.root, "WikiTopics_QE", self.name)  # +raw

    @property
    def processed_dir(self):
        return os.path.join(self.root, "WikiTopics_QE", self.name)  # + processed
    
    @property
    def name(self):
        return f"{self.version}"
    
    def __repr__(self):
        return f"wikitopics_{self.version}"


class InductiveFB15k237QueryExtendedEval(InductiveFB15k237Query):

    """
    This dataset is almost equivalent to the original InductiveComp except that
    validation and test sets are training queries with a new (possibly larger) answer set
    being executed on a bigger validation or test graph

    We will load only the train_queries file and 3 different answer sets:
    1. train_queries_hard - original answers
    2. train_queries_val - answers to train queries over the validation graph (train + new val nodes and edges)
    3. train_queries_test - answers to train queries over the test graph (train + new test nodes and edges)

    The dataset is supposed to be used for evaluation/inference only,
    so make sure num_epochs is set to 0 in the config yaml file
    """

    def load_queries(self, path):
        easy_answers = []
        hard_answers = []
        queries = []
        type_ids = []
        num_samples = []
        num_entity_for_sample = []
        max_query_length = 0

        # in this setup, we evaluate train queries on extended validation/test graphs
        # in extended graphs, training queries now have more answers
        # conceptually, all answers are "easy", but for eval purposes we load them as hard
        with open(os.path.join(path, "train_queries.pkl"), "rb") as fin:
            struct2queries = pickle.load(fin)

        #type2struct = {v: k for k, v in self.struct2type.items()}
        #filtered_training_structs = tuple([type2struct[x] for x in train_patterns])
        for split in ["train", "valid", "test"]:
            if split == "train":
                with open(os.path.join(path, "train_answers_hard.pkl"), "rb") as fin:
                    query2hard_answers = pickle.load(fin)
            else:
                # load new answers
                with open(os.path.join(path, "train_answers_%s.pkl" % split), "rb") as fin:
                    query2hard_answers = pickle.load(fin)

            query2easy_answers = defaultdict(lambda: defaultdict(set))

            num_sample = 0
            structs = sorted(struct2queries.keys(), key=lambda s: self.struct2type[s])
            structs = tqdm(structs, "Loading %s queries" % split)
            for struct in structs:
                query_type = self.struct2type[struct]
                if query_type not in self.type2id:
                    continue

                struct_queries = struct2queries[struct]
                for i, query in enumerate(struct_queries):
                    # The dataset format is slightly different from BetaE's
                    #easy_answers.append(query2easy_answers[struct][i])
                    q_index = i if split != "train" else query
                    hard_answers.append(query2hard_answers[struct][q_index])
                    query = Query.from_nested(query)
                    max_query_length = max(max_query_length, len(query))
                    queries.append(query)
                    type_ids.append(self.type2id[query_type])
                num_sample += len(struct_queries)

            num_entity_for_sample += [getattr(self, "%s_graph" % split).num_nodes] * num_sample
            num_samples.append(num_sample)

        self.queries = queries
        self.types = type_ids

        self.hard_answers = hard_answers
        self.easy_answers = [[] for _ in range(len(hard_answers))]
        self.num_samples = num_samples
        self.num_entity_for_sample = num_entity_for_sample
        self.max_query_length = max_query_length



class JointDataset(LogicalQueryDataset):

    datasets_map = {
        'FB15k237': FB15k237LogicalQuery,
        'FB15k': FB15kLogicalQuery,
        'NELL995': NELL995LogicalQuery,
        # TODO
        'FB_550': partial(InductiveFB15k237Query, version=550),
        'FB_300': partial(InductiveFB15k237Query, version=300),
        'FB_217': partial(InductiveFB15k237Query, version=217),
        'FB_175': partial(InductiveFB15k237Query, version=175),
        'FB_150': partial(InductiveFB15k237Query, version=150),
        'FB_134': partial(InductiveFB15k237Query, version=134),
        'FB_122': partial(InductiveFB15k237Query, version=122),
        'FB_113': partial(InductiveFB15k237Query, version=113),
        'FB_106': partial(InductiveFB15k237Query, version=106),
        # WikiTopics
        'WT_art': partial(WikiTopicsQuery, version="art"),
        'WT_award': partial(WikiTopicsQuery, version="award"),
        'WT_edu': partial(WikiTopicsQuery, version="edu"),
        'WT_health': partial(WikiTopicsQuery, version="health"),
        'WT_infra': partial(WikiTopicsQuery, version="infra"),
        'WT_loc': partial(WikiTopicsQuery, version="loc"),
        'WT_org': partial(WikiTopicsQuery, version="org"),
        'WT_people': partial(WikiTopicsQuery, version="people"),
        'WT_sci': partial(WikiTopicsQuery, version="sci"),
        'WT_sport': partial(WikiTopicsQuery, version="sport"),
        'WT_tax': partial(WikiTopicsQuery, version="tax"),
    }

    def __init__(self, path, graphs, query_types=None, union_type="DNF"):
        # super(JointDataset, self).__init__(*args, **kwargs)

        # Initialize all specified datasets
        self.graphs = [self.datasets_map[dataset](path=path, query_types=query_types, union_type=union_type) for dataset in graphs.split(',')]
        self.graph_names = graphs

        # Total number of samples obtained from iterating over all graphs
        self.num_samples = [sum(k) for k in zip(*[graph.num_samples for graph in self.graphs])]
        self.valid_samples = [torch.cumsum(torch.tensor(k).flatten(), dim=0) for k in zip([graph.num_samples for graph in self.graphs])]

    def __getitem__(self, index):
        # send a dummy entry, we'll be sampling edges in the collator function
        return torch.zeros(1,1)

    def __len__(self):
        return sum([graph.queries for graph in self.graphs])

    def split(self):
        splits = [[],[],[]]
        for graph in self.graphs:
            offset = 0
            for i, num_sample in enumerate(graph.num_samples):
                split = torch_data.Subset(graph, range(offset, offset + num_sample))
                splits[i].append(split)
                offset += num_sample
        return splits    
    
    @property
    def num_nodes(self):
        """Number of entities in the joint graph"""
        return sum(graph.train_graph.num_nodes for graph in self.graphs)

    @property
    def num_edges(self):
        """Number of edges in the joint graph"""
        return sum(graph.train_graph.num_edges for graph in self.graphs)

    @property
    def num_relations(self):
        """Number of relations in the joint graph"""
        return sum(graph.train_graph.num_relations for graph in self.graphs)



