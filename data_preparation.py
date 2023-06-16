import torch
import math
import torch.nn as nn
import numpy as np
import dill
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import sklearn.metrics
import pandas as pd
from random import choice, sample
from itertools import combinations
import networkx as nx
from torch_sparse import SparseTensor
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkLoader, LinkNeighborLoader, NeighborLoader
import torch_geometric.transforms as T
from torch_geometric.utils.convert import to_networkx
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
from torch_geometric.sampler import NeighborSampler
import os

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
    
    def __call__(self, edgelist_pre, article_to_comm_edge_pre, article_from_news_source_edge_pre): 
        if len(edgelist_pre) < self.min_nodes or len(edgelist_pre) > self.num_samples_total:
            return None, None, None, None, None
        # print(edgelist_pre, article_to_comm_edge_pre, article_from_news_source_edge_pre)
        edgelist = []
        sampled_nodes = set()
        node_mapping = {}
        node_idx = 0
        
        node_mapping, node_idx = self.map_nodes(node_mapping, node_idx,\
             article_to_comm_edge_pre[0], article_to_comm_edge_pre[1])
        article_to_comm_edge = [node_mapping[article_to_comm_edge_pre[0]], node_mapping[article_to_comm_edge_pre[1]]]
        sampled_nodes.add(node_mapping[article_to_comm_edge_pre[1]])
        
        node_mapping, node_idx = self.map_nodes(node_mapping, node_idx,\
            article_from_news_source_edge_pre[1])
        article_from_news_source_edge = [article_from_news_source_edge_pre[0], node_mapping[article_from_news_source_edge_pre[1]]]
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
        sampled_dst = [choice(list(set([x for x in range(self.num_nodes)]) - set(trgs))) for i in range(num_neg_edges)]
        
        for ssrc, sdst in zip(sampled_src, sampled_dst):
            node_mapping, node_idx = self.map_nodes(node_mapping, node_idx, ssrc, sdst)
            edgelist.append([node_mapping[ssrc], node_mapping[sdst]])
            sampled_nodes.add(node_mapping[ssrc])
            sampled_nodes.add(node_mapping[sdst])
            labels.append(0)
        # print(edgelist, labels, list(sampled_nodes), article_to_comm_edge, article_from_news_source_edge)
        # print("---------")
        return edgelist, labels, list(sampled_nodes), article_to_comm_edge, article_from_news_source_edge

def save_processed_data(cached_datapath, training_instances, num_comm_nodes, num_article_nodes, metadata, hetero_validation_insts, hetero_testing_insts):
    with open(cached_datapath, "wb") as fp:
        dill.dump([training_instances, num_comm_nodes, num_article_nodes, metadata, hetero_validation_insts, hetero_testing_insts], fp)

def save_uncomm_tweet_ids(cached_datapath, training_tweet_ids_to_retrieve_uncomm, validation_tweet_ids_to_retrieve_uncomm, testing_tweet_ids_to_retrieve_uncomm):
    with open(cached_datapath, "wb") as fp:
        dill.dump([training_tweet_ids_to_retrieve_uncomm, validation_tweet_ids_to_retrieve_uncomm, testing_tweet_ids_to_retrieve_uncomm], fp)

def load_processed_data(cached_datapath):
    training_instances, num_comm_nodes, num_article_nodes, metadata, hetero_validation_insts, hetero_testing_insts = dill.load(open(cached_datapath, 'rb'))
    
    return training_instances, num_comm_nodes, num_article_nodes, metadata, hetero_validation_insts, hetero_testing_insts

def load_graph_instances(datapath):
    # datapath = "/home/ataylor2/processing_covid_tweets/MIPS/small_05_15_to_05_31_graph_instances/05-16-2020_05-19-2020_instances.pkl"
    training_g_instances = dill.load(open(datapath, 'rb'))
    return training_g_instances

def load_comm_mapping(comm_mapping_datapath):
    comm_mapping = dill.load(open(comm_mapping_datapath, 'rb'))
    return comm_mapping

def load_article_mapping(article_mapping_datapath):
    article_mapping = dill.load(open(article_mapping_datapath, 'rb'))
    return article_mapping

def load_user_features(node_feats_datapath, num_nodes):#, comm_mapping):
    node_feats = torch.rand(num_nodes, 768)
    # node_feats = dill.load(open(node_feats_datapath, 'rb'))
    # missing_nodes = num_nodes - node_feats.shape[0]
    # if missing_nodes > 0:
    #     node_feats = torch.cat((node_feats, torch.rand(missing_nodes, 768)), 0)
    return node_feats

def load_article_features(article_feats_datapath):
    # node_feats = torch.rand(len(article_mapping), 128)
    # article_feats = dill.load(open(article_feats_datapath, 'rb'))
    article_mapping = dill.load(open(article_feats_datapath, 'rb'))
    article_feats = torch.rand(len(article_mapping), 128)
    return article_feats
    
def convert_edgelist_to_tensor(edgelist, shape):
    mat = np.zeros(shape, int)
    for src, trg in edgelist:
        mat[src, trg] = 1
    return mat

def construct_data_instance(g_instances, g_instance, num_nodes, num_articles, comm_feats, article_feats, negative_sampler):
    data_inst = HeteroData()
    
    comm_comm_pos_edges = g_instances[g_instance]["comm_to_comm"]
    article_to_comm_edge = g_instances[g_instance]["article_to_comm"]
    article_from_news_source_edge = g_instances[g_instance]["article_from_news_source"]
   
    edge_set = set()
    for edge in comm_comm_pos_edges:
        edge_set.add(tuple(edge))
    
    num_edges = len(edge_set)
    
    if num_edges < 3:
        return None 
    
    comm_comm_pos_neg_edges, labels, sampled_nodes, article_to_comm_edge, article_from_news_source_edge = negative_sampler(list(edge_set),article_to_comm_edge,article_from_news_source_edge)
    if comm_comm_pos_neg_edges == None:
        return None
    
    data_inst['community'].node_id = torch.arange(len(sampled_nodes))
    data_inst['article'].node_id = torch.arange(num_articles)
    
    try:
        data_inst['community'].x = torch.index_select(comm_feats, 0, torch.tensor(sampled_nodes)) # [num_comms, num_features_comms]
    except:
        return None
    
    article_idx = article_from_news_source_edge[0] #g_instances[g_instance]["article_to_comm"][0] #articlenode
    # print(article_feats.shape, article_idx, article_to_comm_edge, article_from_news_source_edge)
    data_inst['article'].x = article_feats[article_idx] # [num_articles, num_features_articles]

    comm_to_comm_edges = torch.tensor(comm_comm_pos_neg_edges).T 
    # print(g_instances[g_instance]["article_to_comm"])
    # raise KeyboardInterrupt
    article_to_comm_edges = torch.tensor([article_to_comm_edge]).T
    article_from_news_source_edges = torch.tensor([article_from_news_source_edge]).T
    data_inst['article', 'mentioned_by', 'community'].edge_index = article_to_comm_edges # [2, num_edges_article_ment_comm]
    data_inst['article', 'written_by', 'community'].edge_index = article_from_news_source_edges # [2, num_edges_article_auth_comm]
    data_inst = T.ToUndirected()(data_inst)
    data_inst['community', 'interacts_with', 'community'].edge_index = comm_to_comm_edges # [2, num_edges_comm_comm]

    # if len(set(labels)) > 2:
    #     print("pre", set(labels))
    data_inst['community', 'interacts_with', 'community'].edge_label_index = torch.ones((comm_to_comm_edges.shape[1],1),dtype=torch.long)
    data_inst['community', 'interacts_with', 'community'].edge_labels = torch.tensor(labels, dtype=torch.float)

    
    # if len(set(data_inst['community', 'interacts_with', 'community'].edge_labels.tolist())) > 2:
    #     print("post", set(data_inst['community', 'interacts_with', 'community'].edge_labels.tolist()))
    #     print("comp", len(labels), len(data_inst['community', 'interacts_with', 'community'].edge_labels.tolist()))
            
    return data_inst

def prepare_training_data(g_training_instances_datapath, g_testing_instances_datapath, training_comm_mapping_datapath, training_comm_feats_datapath, eval_comm_feats_datapath, training_article_feats_datapath, eval_article_feats_datapath, subgraph_min_size, subgraph_max_size, tt_split, data_subset, cached_datapath, overwrite=False):
    
    if os.path.exists(cached_datapath) and not overwrite:
        return load_processed_data(cached_datapath)
    
    uncomm_tweet_ids_datapath = "/home/ataylor2/processing_covid_tweets/InfoPathwayModel_2/cached_data_uncomm/uncomm_tweet_ids.pkl"
    training_tweet_ids_to_retrieve_uncomm = []
    validation_tweet_ids_to_retrieve_uncomm = []
    testing_tweet_ids_to_retrieve_uncomm = []
    
    training_g_instances = load_graph_instances(g_training_instances_datapath)
    testing_g_instances = load_graph_instances(g_testing_instances_datapath)
    
    num_users = load_comm_mapping(training_comm_mapping_datapath)
    
    training_comm_feats = load_user_features(training_comm_feats_datapath, len(num_users))
    num_nodes = training_comm_feats.shape[0]

    training_article_feats = load_article_features(training_article_feats_datapath)
    num_articles = training_article_feats.shape[0]
    eval_article_feats = load_article_features(eval_article_feats_datapath, )
    eval_comm_feats = load_user_features(eval_comm_feats_datapath, num_nodes)

    hetero_training_insts = []
    hetero_validation_insts = []
    hetero_testing_insts = []
    evaluation_insts = []
    metadata = None
    
    retr_metadata = True
    dataset_len = int(data_subset*len(training_g_instances))
    train_set_len = tt_split * dataset_len
    val_set_len = int((1-tt_split)/2 * dataset_len)
    test_set_len = int((1-tt_split)/2 * dataset_len)
    print(f"Training set length:{train_set_len}")
    print(f"Validation set length:{val_set_len}")
    print(f"Testing set length:{test_set_len}")
    
    print(f"Data subset: {dataset_len}")
    
    training_negative_sampler = NegativeSampler(num_nodes, subgraph_max_size, subgraph_min_size)

    for i, g_instance in enumerate(tqdm(training_g_instances.keys())):
        
        data_inst = construct_data_instance(training_g_instances, g_instance, num_nodes, num_articles, training_comm_feats, training_article_feats, training_negative_sampler)
        
        if data_inst == None:
            continue
        
        if retr_metadata:
            metadata = data_inst.metadata()
            retr_metadata = False
    
        if len(hetero_training_insts) < train_set_len:
            hetero_training_insts.append(data_inst)
        else:
            break
        
        if i % 100 == 0:
            print(f"training data size: {train_set_len}, validation data size: {val_set_len}, testing data size: {val_set_len}")
            # print("training data:", len(hetero_training_insts), "validation data:", len(hetero_validation_insts), "testing data:", len(hetero_testing_insts),"instance:", i)
        
            
    eval_negative_sampler = NegativeSampler(num_nodes, subgraph_max_size, subgraph_min_size) 
    test_set_start = int(len(testing_g_instances)/2)
    print("test set start", test_set_start)
    
    for j, g_instance in enumerate(tqdm(testing_g_instances.keys())):
        
        data_inst = construct_data_instance(testing_g_instances, g_instance, num_nodes, num_articles, eval_comm_feats, eval_article_feats, eval_negative_sampler)

        if data_inst == None:
            continue
        if len(hetero_validation_insts) < val_set_len:
            hetero_validation_insts.append(data_inst) 
        else:
            if len(hetero_testing_insts) > test_set_len:
                break
            hetero_testing_insts.append(data_inst)
        
    print("training data:", len(hetero_training_insts), "validation data:", len(hetero_validation_insts), "testing data:", len(hetero_testing_insts),"instance:", i + j)

    save_processed_data(cached_datapath, hetero_training_insts, num_nodes, num_articles, metadata, hetero_validation_insts, hetero_testing_insts)
    # save_uncomm_tweet_ids(uncomm_tweet_ids_datapath, training_tweet_ids_to_retrieve_uncomm, validation_tweet_ids_to_retrieve_uncomm, testing_tweet_ids_to_retrieve_uncomm)
    
    return hetero_training_insts, num_nodes, num_articles, metadata, hetero_validation_insts, hetero_testing_insts

if __name__ == "__main__":

    global uncomm
    #PREDEF COMM 
    uncomm = False
    subgraph_max_size = 25 #500
    subgraph_min_size = 4
    data_subset = 0.0025#0.010#0.007
    tt_split = 0.8

    # cached_datapath = "/home/ataylor2/processing_covid_tweets/Thunder/cached_data/cached_data_longformer.pkl"
    #Longformer
    # training_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Engagement_Comm/05-25-2020_instances.pkl"
    # evaluation_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Engagement_Comm/05-30-2020_instances.pkl"
    
    # training_comm_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Engagement_Comm/05-25-2020_comm_mapping.pkl"
    
    # training_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_longformer_embedding/05-19-2020_embedding.json"
    # eval_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_longformer_embedding/05-25-2020_embedding.json"
    
    # article_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Engagement_Comm/05-30-2020_article_mapping.pkl"

    # training_article_feats_datapath = article_mapping_datapath
    # eval_article_feats_datapath = article_mapping_datapath
    # # training_article_feats_datapath = "/home/ataylor2/processing_covid_tweets/InfoPathwayModel_2/demo_data/05-20-2020_05-25-2020_article_feats.pkl" #TODO
    # # eval_article_feats_datapath = "/home/ataylor2/processing_covid_tweets/InfoPathwayModel_2/demo_data/05-26-2020_05-30-2020_article_feats.pkl"

    #Primera
    #cached_datapath = "/home/ataylor2/processing_covid_tweets/Thunder/cached_data/cached_data_primera_bert.pkl"

    # training_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Engagement_Comm/05-25-2020_instances.pkl"
    # evaluation_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Engagement_Comm/05-30-2020_instances.pkl"
    
    # training_comm_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Engagement_Comm/05-25-2020_comm_mapping.pkl"
    
    # training_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_primera_bert_embedding/05-19-2020_embedding.json"
    # eval_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_primera_bert_embedding/05-25-2020_embedding.json"
    
    # article_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Engagement_Comm/05-30-2020_article_mapping.pkl"

    # training_article_feats_datapath = article_mapping_datapath
    # eval_article_feats_datapath = article_mapping_datapath
    
    #Primera CLIP
    # cached_datapath = "/home/ataylor2/processing_covid_tweets/Thunder/cached_data/cached_data_primera_clip.pkl"

    # training_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Engagement_Comm/05-25-2020_instances.pkl"
    # evaluation_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Engagement_Comm/05-30-2020_instances.pkl"
    
    # training_comm_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Engagement_Comm/05-25-2020_comm_mapping.pkl"
    
    # training_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_primera_clip_embedding/05-19-2020_embedding.json"
    # eval_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_primera_clip_embedding/05-25-2020_embedding.json"
    
    # article_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Engagement_Comm/05-30-2020_article_mapping.pkl"

    # training_article_feats_datapath = article_mapping_datapath
    # eval_article_feats_datapath = article_mapping_datapath
    
    ###############################################################################################################################################
    
    #Longformer
    # cached_datapath = "/home/ataylor2/processing_covid_tweets/Thunder/cached_data/cached_int_longformer.pkl"

    # training_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Interaction_Comm/05-25-2020_instances.pkl"
    # evaluation_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Interaction_Comm/05-30-2020_instances.pkl"
    
    # training_comm_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Interaction_Comm/05-25-2020_comm_mapping.pkl"
    
    # training_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_longformer_embedding/05-19-2020_embedding.json"
    # eval_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_longformer_embedding/05-25-2020_embedding.json"
    
    # article_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Interaction_Comm/05-30-2020_article_mapping.pkl"

    # training_article_feats_datapath = article_mapping_datapath
    # eval_article_feats_datapath = article_mapping_datapath
    # # training_article_feats_datapath = "/home/ataylor2/processing_covid_tweets/InfoPathwayModel_2/demo_data/05-20-2020_05-25-2020_article_feats.pkl" #TODO
    # # eval_article_feats_datapath = "/home/ataylor2/processing_covid_tweets/InfoPathwayModel_2/demo_data/05-26-2020_05-30-2020_article_feats.pkl"

    #Primera BERT
    # cached_datapath = "/home/ataylor2/processing_covid_tweets/Thunder/cached_data/cached_int_primera_bert.pkl"

    # training_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Interaction_Comm/05-25-2020_instances.pkl"
    # evaluation_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Interaction_Comm/05-30-2020_instances.pkl"
    
    # training_comm_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Interaction_Comm/05-25-2020_comm_mapping.pkl"
    
    # training_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_primera_bert_embedding/05-19-2020_embedding.json"
    # eval_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_primera_bert_embedding/05-25-2020_embedding.json"
    
    # article_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Interaction_Comm/05-30-2020_article_mapping.pkl"

    # training_article_feats_datapath = article_mapping_datapath
    # eval_article_feats_datapath = article_mapping_datapath
    
    #Primera CLIP
    # cached_datapath = "/home/ataylor2/processing_covid_tweets/Thunder/cached_data/cached_int_primera_clip.pkl"

    # training_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Interaction_Comm/05-25-2020_instances.pkl"
    # evaluation_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Interaction_Comm/05-30-2020_instances.pkl"
    
    # training_comm_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Interaction_Comm/05-25-2020_comm_mapping.pkl"
    
    # training_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_primera_clip_embedding/05-19-2020_embedding.json"
    # eval_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_primera_clip_embedding/05-25-2020_embedding.json"
    
    # article_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Interaction_Comm/05-30-2020_article_mapping.pkl"

    # training_article_feats_datapath = article_mapping_datapath
    # eval_article_feats_datapath = article_mapping_datapath
    
    ###############################################################################################################################################

    #Longformer
    cached_datapath = "/home/ataylor2/processing_covid_tweets/Thunder/cached_data/cached_ip_longformer.pkl"

    training_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/IP_Comm/05-25-2020_instances.pkl"
    evaluation_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/IP_Comm/05-30-2020_instances.pkl"
    
    training_comm_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/IP_Comm/05-25-2020_comm_mapping.pkl"
    
    training_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_longformer_embedding/05-19-2020_embedding.json"
    eval_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_longformer_embedding/05-25-2020_embedding.json"
    
    article_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/IP_Comm/05-30-2020_article_mapping.pkl"

    training_article_feats_datapath = article_mapping_datapath
    eval_article_feats_datapath = article_mapping_datapath
    prepare_training_data(training_g_instances_datapath, evaluation_g_instances_datapath, training_comm_mapping_datapath, training_comm_feats_datapath, eval_comm_feats_datapath, training_article_feats_datapath, eval_article_feats_datapath, subgraph_min_size, subgraph_max_size, tt_split, data_subset, cached_datapath, overwrite=True)
  
    #Primera BERT
    cached_datapath = "/home/ataylor2/processing_covid_tweets/Thunder/cached_data/cached_ip_primera_bert.pkl"

    training_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/IP_Comm/05-25-2020_instances.pkl"
    evaluation_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/IP_Comm/05-30-2020_instances.pkl"
    
    training_comm_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/IP_Comm/05-25-2020_comm_mapping.pkl"
    
    training_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_primera_bert_embedding/05-19-2020_embedding.json"
    eval_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_primera_bert_embedding/05-25-2020_embedding.json"
    
    article_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/IP_Comm/05-30-2020_article_mapping.pkl"
    
    training_article_feats_datapath = article_mapping_datapath
    eval_article_feats_datapath = article_mapping_datapath
    
    prepare_training_data(training_g_instances_datapath, evaluation_g_instances_datapath, training_comm_mapping_datapath, training_comm_feats_datapath, eval_comm_feats_datapath, training_article_feats_datapath, eval_article_feats_datapath, subgraph_min_size, subgraph_max_size, tt_split, data_subset, cached_datapath, overwrite=True)
    
    #Primera CLIP
    cached_datapath = "/home/ataylor2/processing_covid_tweets/Thunder/cached_data/cached_ip_primera_clip.pkl"

    training_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/IP_Comm/05-25-2020_instances.pkl"
    evaluation_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/IP_Comm/05-30-2020_instances.pkl"
    
    training_comm_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/IP_Comm/05-25-2020_comm_mapping.pkl"
    
    training_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_primera_clip_embedding/05-19-2020_embedding.json"
    eval_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_primera_clip_embedding/05-25-2020_embedding.json"
    
    article_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/IP_Comm/05-30-2020_article_mapping.pkl"

    training_article_feats_datapath = article_mapping_datapath
    eval_article_feats_datapath = article_mapping_datapath
    
    prepare_training_data(training_g_instances_datapath, evaluation_g_instances_datapath, training_comm_mapping_datapath, training_comm_feats_datapath, eval_comm_feats_datapath, training_article_feats_datapath, eval_article_feats_datapath, subgraph_min_size, subgraph_max_size, tt_split, data_subset, cached_datapath, overwrite=True)
