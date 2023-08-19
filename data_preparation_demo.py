import torch
import math
import torch.nn as nn
import numpy as np
import dill
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F
import random
from random import choice, sample
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import os
import argparse
import json
import utils

class NegativeSampler(object):
    def __init__(self, num_comm_nodes, subgraph_max_size, subgraph_min_size):
        self.num_samples_total = subgraph_max_size
        self.min_nodes = subgraph_min_size
        self.num_nodes = num_comm_nodes
    
    def map_nodes(self, node_mapping, node_idx, source, target=None):
        if not source in node_mapping:
            node_mapping[source] = node_idx
            node_idx += 1
        if target != None:
            if not target in node_mapping:
                node_mapping[target] = node_idx
                node_idx += 1
        return node_mapping, node_idx
    
    def __call__(self, edgelist_pre, article_to_comm_edge_pre, article_from_news_source_edge_pre, community_mapping): 
        if len(edgelist_pre) < self.min_nodes or len(edgelist_pre) > self.num_samples_total:
            return None, None, None, None, None, None
        
        edgelist = []
        sampled_nodes = set()
        node_mapping = {}
        # node_idx = 0

        # node_mapping, node_idx = self.map_nodes(node_mapping, node_idx,\
        #      article_to_comm_edge_pre[0], article_to_comm_edge_pre[1])
        # article_to_comm_edge = [node_mapping[article_to_comm_edge_pre[0]], node_mapping[article_to_comm_edge_pre[1]]]
        # sampled_nodes.add(node_mapping[article_to_comm_edge_pre[1]])
        
        # node_mapping, node_idx = self.map_nodes(node_mapping, node_idx,\
        #     article_from_news_source_edge_pre[1])
        # article_from_news_source_edge = [node_mapping[article_from_news_source_edge_pre[0]], node_mapping[article_from_news_source_edge_pre[1]]]
        # sampled_nodes.add(node_mapping[article_from_news_source_edge_pre[1]])
        
        node_idx = 0

        node_mapping, node_idx = self.map_nodes(node_mapping, node_idx, article_to_comm_edge_pre[1])
        article_to_comm_edge = [0, node_mapping[article_to_comm_edge_pre[1]]]
        sampled_nodes.add(node_mapping[article_to_comm_edge_pre[1]])
        
        node_mapping, node_idx = self.map_nodes(node_mapping, node_idx,\
            article_from_news_source_edge_pre[1])
        article_from_news_source_edge = [0, node_mapping[article_from_news_source_edge_pre[1]]]
        sampled_nodes.add(node_mapping[article_from_news_source_edge_pre[1]])
        
        
        srcs = []
        trgs = []
        for edge in edgelist_pre:
            srcs.append(edge[0])
            trgs.append(edge[1])

            node_mapping, node_idx = self.map_nodes(node_mapping, node_idx, edge[0], edge[1])
            
            sampled_nodes.add(node_mapping[edge[0]])
            sampled_nodes.add(node_mapping[edge[1]])
            
            edgelist.append([node_mapping[edge[0]], node_mapping[edge[1]]])
            
        num_neg_edges = self.num_samples_total - len(edgelist)
        labels = [1] * len(edgelist)
   
        sampled_src = [choice(srcs) for _ in range(num_neg_edges)]
        sampled_dst = [choice(list(set(list(community_mapping.keys())) - set(trgs))) for i in range(num_neg_edges)]
        
        for ssrc, sdst in zip(sampled_src, sampled_dst):
            node_mapping, node_idx = self.map_nodes(node_mapping, node_idx, ssrc, sdst)
            edgelist.append([node_mapping[ssrc], node_mapping[sdst]])
            sampled_nodes.add(node_mapping[ssrc])
            sampled_nodes.add(node_mapping[sdst])
            labels.append(0)

        return edgelist, labels, list(sampled_nodes), article_to_comm_edge, article_from_news_source_edge, node_mapping

def save_processed_data(cached_datapath, training_instances, num_comm_nodes, num_article_nodes, metadata, hetero_validation_insts, hetero_testing_insts):
    with open(cached_datapath, "wb") as fp:
        dill.dump([training_instances, num_comm_nodes, num_article_nodes, metadata, hetero_validation_insts, hetero_testing_insts], fp)
        
def load_processed_data(cached_datapath):
    training_instances, num_comm_nodes, num_article_nodes, metadata, hetero_validation_insts, hetero_testing_insts = dill.load(open(cached_datapath, 'rb'))
    
    return training_instances, num_comm_nodes, num_article_nodes, metadata, hetero_validation_insts, hetero_testing_insts

    
def convert_edgelist_to_tensor(edgelist, shape):
    mat = np.zeros(shape, int)
    for src, trg in edgelist:
        mat[src, trg] = 1
    return mat

def construct_data_instance(g_instances, g_instance, num_nodes, num_articles, comm_feats, comm_mapping, article_feats, negative_sampler):
    data_inst = HeteroData()
    
    # comm_comm_pos_edges = g_instances[g_instance]["comm_to_comm"]
    article_to_comm_edge = g_instances[g_instance]["article_to_comm"]
    article_from_news_source_edge = g_instances[g_instance]["article_from_news_source"]
    comm_to_comm_with_neg_labels = g_instances[g_instance]["comm_to_comm_with_neg_labels"]
    comm_to_comm_with_neg = g_instances[g_instance]["comm_to_comm_with_neg"]
    # comm_path_to_user_path_mapping = g_instances[g_instance]['comm_path_to_user_path_mapping']
    user_level_pathway_with_neg = g_instances[g_instance]['user_level_pathway_with_neg']
    user_level_pathway_with_neg_labels = g_instances[g_instance]['user_level_pathway_with_neg_labels']
    
    # TODO: temporary fix: logically correct
    article_to_comm_edge[0] = article_from_news_source_edge[0]
    article_idx = article_from_news_source_edge[0]
    
    comm_comm_pos_neg_edges = g_instances[g_instance]["comm_to_comm_with_neg"]
    
    community_idx_mapping = utils.create_mapping(article_to_comm_edge, edge_type="article")
    community_idx_mapping = utils.create_mapping(article_from_news_source_edge, existing_mapping=community_idx_mapping, edge_type="article")
    community_idx_mapping = utils.create_mapping(comm_comm_pos_neg_edges, existing_mapping=community_idx_mapping)
    article_to_comm_edge = utils.execute_mapping(article_to_comm_edge, community_idx_mapping, edge_type="article")
    article_from_news_source_edge = utils.execute_mapping(article_from_news_source_edge, community_idx_mapping, edge_type="article")
    comm_comm_pos_neg_edges = utils.execute_mapping(comm_comm_pos_neg_edges, community_idx_mapping)
    
    if len(comm_comm_pos_neg_edges) < 3:
        return None 
    
    # comm_comm_pos_neg_edges, labels, sampled_nodes, article_to_comm_edge, article_from_news_source_edge, community_idx_mapping = negative_sampler(list(comm_comm_pos_edges),article_to_comm_edge,article_from_news_source_edge, comm_mapping)
    
    if comm_comm_pos_neg_edges == None:
        return None
    data_inst['key'] = g_instance
    data_inst['community'].node_id = torch.arange(len(community_idx_mapping))
    # TODO: temporary fix: logically correct
    data_inst['article'].node_id = torch.arange(1)
    try:
        data_inst['community'].x = utils.merge_tensor_dicts(community_idx_mapping, comm_feats)
    except KeyError:
        return None
    
    try:
        data_inst['article'].x = article_feats[article_idx] 
    except KeyError:
        return None
    #     data_inst['article'].x = article_feats[list(article_feats.keys())[random.randint(0, len(article_feats))]]  

    comm_to_comm_edges = torch.tensor(comm_comm_pos_neg_edges).T 

    article_to_comm_edges = torch.tensor([article_to_comm_edge]).T
    article_from_news_source_edges = torch.tensor([article_from_news_source_edge]).T
    data_inst['article', 'mentioned_by', 'community'].edge_index = article_to_comm_edges # [2, num_edges_article_ment_comm]
    data_inst['article', 'written_by', 'community'].edge_index = article_from_news_source_edges # [2, num_edges_article_auth_comm]
    data_inst = T.ToUndirected()(data_inst)

    data_inst['community', 'interacts_with', 'community'].edge_index = comm_to_comm_edges # [2, num_edges_comm_comm]
    data_inst['community', 'interacts_with', 'community'].edge_label_index = torch.ones((comm_to_comm_edges.shape[1],1),dtype=torch.long)
    data_inst['community', 'interacts_with', 'community'].edge_labels = torch.tensor(comm_to_comm_with_neg_labels, dtype=torch.float)
    data_inst['community', 'interacts_with', 'community'].edge_labels_str = comm_to_comm_with_neg
    # data_inst['community', 'interacts_with', 'community'].comm_user_map = comm_path_to_user_path_mapping
    data_inst['community', 'interacts_with', 'community'].user_edge_labels_str = user_level_pathway_with_neg
    data_inst['community', 'interacts_with', 'community'].user_edge_labels = user_level_pathway_with_neg_labels
            
    return data_inst

def prepare_training_data(training_instances_path, evaluation_instances_path, training_community_features_path, evaluation_community_features_path, training_article_features_path, evaluation_article_features_path, subgraph_min_size, subgraph_max_size, cached_datapath, overwrite=False, demo_data_gen=False):
    
    if os.path.exists(cached_datapath) and not overwrite:
        return load_processed_data(cached_datapath)
    
    training_g_instances = utils.load_dill(training_instances_path)
    testing_g_instances = utils.load_dill(evaluation_instances_path)
    
    training_comm_feats = utils.load_dill(training_community_features_path)
    training_comm_mapping = utils.create_key_mapping(training_comm_feats)
    num_nodes = len(training_comm_feats)
    training_article_feats = utils.load_dill(training_article_features_path)
    num_articles = len(training_article_feats)
    
    eval_comm_feats = utils.load_dill(evaluation_community_features_path)
    eval_comm_mapping = utils.create_key_mapping(eval_comm_feats)
    eval_article_feats = utils.load_dill(evaluation_article_features_path)

    hetero_training_insts = []
    evaluation_insts = []
    metadata = None
    
    retr_metadata = True
    
    training_negative_sampler = NegativeSampler(num_nodes, subgraph_max_size, subgraph_min_size)

    for i, g_instance in enumerate(tqdm(training_g_instances.keys())):
        data_inst = construct_data_instance(training_g_instances, g_instance, num_nodes, num_articles, training_comm_feats, training_comm_mapping, training_article_feats, training_negative_sampler)
        
        if data_inst == None:
            continue
        
        if retr_metadata:
            metadata = data_inst.metadata()
            retr_metadata = False

        hetero_training_insts.append(data_inst)
        
    print(f"training data size: {len(hetero_training_insts)}, validation data size: {len(evaluation_insts)}, testing data size: {len(evaluation_insts)}")        

    eval_negative_sampler = NegativeSampler(num_nodes, subgraph_max_size, subgraph_min_size) 
    val_set_len = int(len(testing_g_instances)/2)
    
    for j, g_instance in enumerate(tqdm(testing_g_instances.keys())):
        
        data_inst = construct_data_instance(testing_g_instances, g_instance, num_nodes, num_articles, eval_comm_feats, eval_comm_mapping, eval_article_feats, eval_negative_sampler)

        if data_inst == None:
            continue
        evaluation_insts.append(data_inst) 

    midpoint = len(evaluation_insts) // 2
    hetero_validation_insts = evaluation_insts[:midpoint]
    hetero_testing_insts = evaluation_insts[midpoint:]
    
    if demo_data_gen:
        demo_insts = []
        demo_insts.extend(hetero_training_insts)
        demo_insts.extend(evaluation_insts)
        return demo_insts
        
    print("training data:", len(hetero_training_insts), "validation data:", len(hetero_validation_insts), "testing data:", len(hetero_testing_insts),"instance:", i + j)

    save_processed_data(cached_datapath, hetero_training_insts, num_nodes, num_articles, metadata, hetero_validation_insts, hetero_testing_insts)
    
    return hetero_training_insts, num_nodes, num_articles, metadata, hetero_validation_insts, hetero_testing_insts

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Read configuration JSON file')
    parser.add_argument('config', type=str, help='Path to the configuration JSON file')
    args = parser.parse_args()

    config = utils.read_config(args.config)
    config.update(args.__dict__)
    config = argparse.Namespace(**config)
    
    cached_datapath = os.path.join(config.cached_datapath,f"cached_{config.community_aggregation_type}_{config.community_embedding_model_name}.pkl") 
    training_instances_path = os.path.join(config.info_pathway_instances_datapath, config.training_instances_filename)
    evaluation_instances_path = os.path.join(config.info_pathway_instances_datapath, config.training_instances_filename)
    
    training_community_features_path = os.path.join(config.comm_feats_datapath, config.training_comm_features)
    evaluation_community_features_path = os.path.join(config.comm_feats_datapath, config.eval_comm_features)
    
    training_article_features_path = os.path.join(config.article_features_datapath, config.training_article_features)
    evaluation_article_features_path = os.path.join(config.article_features_datapath, config.eval_article_features)
    
    prepare_training_data(training_instances_path, evaluation_instances_path, training_community_features_path, evaluation_community_features_path, training_article_features_path, evaluation_article_features_path, config.subgraph_min_size, config.subgraph_max_size, cached_datapath, overwrite=True, demo_data_gen=config.demo_data_generation)
