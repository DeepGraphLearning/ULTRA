import os
import sys
import csv
import math
import time
import pprint
import argparse
import random

import torch
import torch_geometric as pyg
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from torch.utils import data as torch_data
from torch_geometric.data import Data

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ultra import tasks, util
from ultra.models import Ultra
from script.run import train_and_validate, test


default_finetuning_config = {
    # graph: (num_epochs, batches_per_epoch), null means all triples in train set
    # transductive datasets (17)
    # standard ones (10)
    "CoDExSmall": (1, 4000),
    "CoDExMedium": (1, 4000),
    "CoDExLarge": (1, 2000),
    "FB15k237": (1, 'null'),
    "WN18RR": (1, 'null'),
    "YAGO310": (1, 2000),
    "DBpedia100k": (1, 1000),
    "AristoV4": (1, 2000),
    "ConceptNet100k": (1, 2000),
    "ATOMIC": (1, 200),
    # tail-only datasets (2)
    "NELL995": (1, 'null'),  # not implemented yet
    "Hetionet": (1, 4000),
    # sparse datasets (5)
    "WDsinger": (3, 'null'),
    "FB15k237_10": (1, 'null'),
    "FB15k237_20": (1, 'null'),
    "FB15k237_50": (1, 1000),
    "NELL23k": (3, 'null'),
    # inductive datasets (42)
    # GraIL datasets (12)
    "FB15k237Inductive": (1, 'null'),    # for all 4 datasets
    "WN18RRInductive": (1, 'null'),      # for all 4 datasets
    "NELLInductive": (3, 'null'),        # for all 4 datasets
    # ILPC (2)
    "ILPC2022SmallInductive": (3, 'null'),
    "ILPC2022LargeInductive": (1, 1000),
    # Ingram datasets (13)
    "NLIngram": (3, 'null'),  # for all 5 datasets
    "FBIngram": (3, 'null'),  # for all 4 datasets
    "WKIngram": (3, 'null'),  # for all 4 datasets
    # MTDEA datasets (10)
    "WikiTopicsMT1": (3, 'null'),  # for all 2 test datasets
    "WikiTopicsMT2": (3, 'null'),  # for all 2 test datasets
    "WikiTopicsMT3": (3, 'null'),  # for all 2 test datasets
    "WikiTopicsMT4": (3, 'null'),  # for all 2 test datasets
    "Metafam": (3, 'null'),
    "FBNELL": (3, 'null'),
    # Hamaguchi datasets (4)
    "HM": (1, 100)  # for all 4 datasets
}

default_train_config = {
    # graph: (num_epochs, batches_per_epoch), null means all triples in train set
    # transductive datasets (17)
    # standard ones (10)
    "CoDExSmall": (10, 1000),
    "CoDExMedium": (10, 1000),
    "CoDExLarge": (10, 1000),
    "FB15k237": (10, 1000),
    "WN18RR": (10, 1000),
    "YAGO310": (10, 2000),
    "DBpedia100k": (10, 1000),
    "AristoV4": (10, 1000),
    "ConceptNet100k": (10, 1000),
    "ATOMIC": (10, 1000),
    # tail-only datasets (2)
    "NELL995": (10, 1000),  # not implemented yet
    "Hetionet": (10, 1000),
    # sparse datasets (5)
    "WDsinger": (10, 1000),
    "FB15k237_10": (10, 1000),
    "FB15k237_20": (10, 1000),
    "FB15k237_50": (10, 1000),
    "NELL23k": (10, 1000),
    # inductive datasets (42)
    # GraIL datasets (12)
    "FB15k237Inductive": (10, 'null'),    # for all 4 datasets
    "WN18RRInductive": (10, 'null'),      # for all 4 datasets
    "NELLInductive": (10, 'null'),        # for all 4 datasets
    # ILPC (2)
    "ILPC2022SmallInductive": (10, 'null'),
    "ILPC2022LargeInductive": (10, 1000),
    # Ingram datasets (13)
    "NLIngram": (10, 'null'),  # for all 5 datasets
    "FBIngram": (10, 'null'),  # for all 4 datasets
    "WKIngram": (10, 'null'),  # for all 4 datasets
    # MTDEA datasets (10)
    "WikiTopicsMT1": (10, 'null'),  # for all 2 test datasets
    "WikiTopicsMT2": (10, 'null'),  # for all 2 test datasets
    "WikiTopicsMT3": (10, 'null'),  # for all 2 test datasets
    "WikiTopicsMT4": (10, 'null'),  # for all 2 test datasets
    "Metafam": (10, 'null'),
    "FBNELL": (10, 'null'),
    # Hamaguchi datasets (4)
    "HM": (10, 1000)  # for all 4 datasets
}


separator = ">" * 30
line = "-" * 30

def set_seed(seed):
    random.seed(seed + util.get_rank())
    # np.random.seed(seed + util.get_rank())
    torch.manual_seed(seed + util.get_rank())
    torch.cuda.manual_seed(seed + util.get_rank())
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":

    seeds = [1024, 42, 1337, 512, 256]

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-d", "--datasets", help="target datasets", default='FB15k237Inductive:v1,NELLInductive:v4', type=str, required=True)
    parser.add_argument("-reps", "--repeats", help="number of times to repeat each exp", default=1, type=int)
    parser.add_argument("-ft", "--finetune", help="finetune the checkpoint on the specified datasets", action='store_true')
    parser.add_argument("-tr", "--train", help="train the model from scratch", action='store_true')
    args, unparsed = parser.parse_known_args()
   
    datasets = args.datasets.split(",")
    path = os.path.dirname(os.path.expanduser(__file__))
    results_file = os.path.join(path, f"ultra_results_{time.strftime('%Y-%m-%d-%H-%M-%S')}.csv")

    for graph in datasets:
        ds, version = graph.split(":") if ":" in graph else (graph, None)
        for i in range(args.repeats):
            seed = seeds[i] if i < len(seeds) else random.randint(0, 10000)
            print(f"Running on {graph}, iteration {i+1} / {args.repeats}, seed: {seed}")

            # get dynamic arguments defined in the config file
            vars = util.detect_variables(args.config)
            parser = argparse.ArgumentParser()
            for var in vars:
                parser.add_argument("--%s" % var)
            vars = parser.parse_known_args(unparsed)[0]
            vars = {k: util.literal_eval(v) for k, v in vars._get_kwargs()}

            if args.finetune:
                epochs, batch_per_epoch = default_finetuning_config[ds] 
            elif args.train:
                epochs, batch_per_epoch = default_train_config[ds] 
            else:
                epochs, batch_per_epoch = 0, 'null'
            vars['epochs'] = epochs
            vars['bpe'] = batch_per_epoch
            vars['dataset'] = ds
            if version is not None:
                vars['version'] = version
            cfg = util.load_config(args.config, context=vars)

            root_dir = os.path.expanduser(cfg.output_dir) # resetting the path to avoid inf nesting
            os.chdir(root_dir)
            working_dir = util.create_working_directory(cfg)
            set_seed(seed)

            # args, vars = util.parse_args()
            # cfg = util.load_config(args.config, context=vars)
            # working_dir = util.create_working_directory(cfg)
            # torch.manual_seed(args.seed + util.get_rank())
            logger = util.get_root_logger()
            if util.get_rank() == 0:
                logger.warning("Random seed: %d" % seed)
                logger.warning("Config file: %s" % args.config)
                logger.warning(pprint.pformat(cfg))
            
            task_name = cfg.task["name"]
            dataset = util.build_dataset(cfg)
            device = util.get_device(cfg)
            
            train_data, valid_data, test_data = dataset[0], dataset[1], dataset[2]
            train_data = train_data.to(device)
            valid_data = valid_data.to(device)
            test_data = test_data.to(device)

            model = Ultra(
                rel_model_cfg=cfg.model.relation_model,
                entity_model_cfg=cfg.model.entity_model,
            )

            if "checkpoint" in cfg and cfg.checkpoint is not None:
                state = torch.load(cfg.checkpoint, map_location="cpu")
                model.load_state_dict(state["model"])

            #model = pyg.compile(model, dynamic=True)
            model = model.to(device)
            
            if task_name == "InductiveInference":
                # filtering for inductive datasets
                # Grail, MTDEA, HM datasets have validation sets based off the training graph
                # ILPC, Ingram have validation sets from the inference graph
                # filtering dataset should contain all true edges (base graph + (valid) + test) 
                if "ILPC" in cfg.dataset['class'] or "Ingram" in cfg.dataset['class']:
                    # add inference, valid, test as the validation and test filtering graphs
                    full_inference_edges = torch.cat([valid_data.edge_index, valid_data.target_edge_index, test_data.target_edge_index], dim=1)
                    full_inference_etypes = torch.cat([valid_data.edge_type, valid_data.target_edge_type, test_data.target_edge_type])
                    test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)
                    val_filtered_data = test_filtered_data
                else:
                    # test filtering graph: inference edges + test edges
                    full_inference_edges = torch.cat([test_data.edge_index, test_data.target_edge_index], dim=1)
                    full_inference_etypes = torch.cat([test_data.edge_type, test_data.target_edge_type])
                    test_filtered_data = Data(edge_index=full_inference_edges, edge_type=full_inference_etypes, num_nodes=test_data.num_nodes)

                    # validation filtering graph: train edges + validation edges
                    val_filtered_data = Data(
                        edge_index=torch.cat([train_data.edge_index, valid_data.target_edge_index], dim=1),
                        edge_type=torch.cat([train_data.edge_type, valid_data.target_edge_type])
                    )
                #test_filtered_data = val_filtered_data = None
            else:
                # for transductive setting, use the whole graph for filtered ranking
                filtered_data = Data(edge_index=dataset._data.target_edge_index, edge_type=dataset._data.target_edge_type, num_nodes=dataset[0].num_nodes)
                val_filtered_data = test_filtered_data = filtered_data
            
            val_filtered_data = val_filtered_data.to(device)
            test_filtered_data = test_filtered_data.to(device)
            
            train_and_validate(cfg, model, train_data, valid_data, filtered_data=val_filtered_data, device=device, logger=logger)
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Evaluate on valid")
            test(cfg, model, valid_data, filtered_data=val_filtered_data, device=device, logger=logger)
            if util.get_rank() == 0:
                logger.warning(separator)
                logger.warning("Evaluate on test")
            metrics = test(cfg, model, test_data, filtered_data=test_filtered_data, return_metrics=True, device=device, logger=logger)

            metrics = {k:v.item() for k,v in metrics.items()}
            metrics['dataset'] = graph
            # write to the log file
            with open(results_file, "a", newline='') as csv_file:
                fieldnames = ['dataset']+list(metrics.keys())[:-1]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=',')
                if csv_file.tell() == 0:
                    writer.writeheader()
                writer.writerow(metrics)