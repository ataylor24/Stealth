import os
import torch
import torch.nn.functional as F
import networkx as nx
import random
import torch_geometric.transforms as T
from torch import Tensor
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, convert, to_dense_adj
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn.conv import MessagePassing
# from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from data_preparation import prepare_training_data
import traceback
from tqdm import tqdm
from model import Model
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import torch.optim as optim
from statistics import mean

device = 'cuda:4'

def information_pathway_evaluation(preds, ground_truths):
    agg_acc = []
    for pred,ground_truth in zip(preds, ground_truths):
        agg_acc.append(accuracy_score(ground_truth.cpu().numpy().astype(int), pred.cpu().numpy().astype(int)))

    return mean(agg_acc)

def run_iteration(model, sampled_data):
    sampled_data#.to(device)
    
    print("memory used", torch.cuda.memory_allocated('cuda:6'))
    # sampled_data.detach()
    print("memory used", torch.cuda.memory_allocated('cuda:6'))
    print("--------------------")
    # torch.cuda.empty_cache()
    # pred = model(sampled_data)
    # ground_truth = sampled_data['community', 'interacts_with', 'community'].edge_label
    # loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
    # loss.backward()

def main():
    model_output_path = "/home/ataylor2/processing_covid_tweets/Thunder/Best_Models"

    cached_datapath = "/home/ataylor2/processing_covid_tweets/Thunder/cached_data/cached_ip_primera_clip.pkl"
    
    training_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Engagement_Comm/05-25-2020_instances.pkl"
    evaluation_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Engagement_Comm/05-30-2020_instances.pkl"
    
    training_comm_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Engagement_Comm/05-25-2020_comm_mapping.pkl"
    
    training_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_longformer_embedding/05-19-2020_embedding.json"
    eval_comm_feats_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_longformer_embedding/05-25-2020_embedding.json"
    
    article_mapping_datapath = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Information_Pathways/Engagement_Comm/05-25-2020_article_mapping.pkl"

    training_article_feats_datapath = None
    eval_article_feats_datapath = None
    # test_g_instances_datapath = "/home/ataylor2/processing_covid_tweets/InfoPathwayModel_2/heterogeneous_data_small/05-20-2020_05-25-2020_instances.pkl"    
    # test_comm_mapping_datapath = "/home/ataylor2/processing_covid_tweets/InfoPathwayModel_2/heterogeneous_data_small/05-20-2020_05-25-2020_comm_mapping.pkl"
    # test_article_mapping_datapath = "/home/ataylor2/processing_covid_tweets/InfoPathwayModel_2/heterogeneous_data_small/05-20-2020_05-25-2020_article_mapping.pkl"
    
    # training_article_feats_datapath = "/home/ataylor2/processing_covid_tweets/InfoPathwayModel_2/heterogeneous_data/05-20-2020_05-25-2020_article_feats.pkl" #TODO
    # eval_article_feats_datapath = "/home/ataylor2/processing_covid_tweets/InfoPathwayModel_2/heterogeneous_data/05-26-2020_05-30-2020_article_feats.pkl"
    
    
    
    
    subgraph_max_size = 25 #500
    subgraph_min_size = 4
    data_subset = 0.010#0.007
    tt_split = 0.8
    modelname = "gat" #gat or graphsage
    out_channels = 25#750#3#1 #32one less
    
    best_micro_f1 = -1
    best_roc_auc = -1
    test_results = {}
    
    training_instances, num_comm_nodes, num_article_nodes, metadata, \
        hetero_validation_insts, hetero_testing_insts = prepare_training_data(training_g_instances_datapath, \
            evaluation_g_instances_datapath, training_comm_mapping_datapath, training_comm_feats_datapath, \
        eval_comm_feats_datapath, training_article_feats_datapath, eval_article_feats_datapath, subgraph_min_size, subgraph_max_size, tt_split, data_subset, cached_datapath)    
    
    print("--------------------------")
    bl_ground_truths = []
    for sampled_data in tqdm(hetero_testing_insts):
        # print(sampled_data['community', 'interacts_with', 'community'])
        bl_ground_truths.append(sampled_data['community', 'interacts_with', 'community'].edge_labels)
    
    bl_agg_gts = torch.cat(bl_ground_truths, dim=0).cpu().tolist()
    print("BL Random",sum(bl_agg_gts)/len(bl_agg_gts))
    print("--------------------------")
    
    print("training_instances", len(training_instances))
    model = Model(hidden_channels=subgraph_max_size, out_channels=out_channels, modelname=modelname, num_comm_nodes=num_comm_nodes, num_article_nodes=num_article_nodes, metadata=metadata)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=40)
 
    
    model.train()
    for epoch in range(1, 40):
        total_loss = total_examples = 0
        for inst_num, sampled_data in enumerate(tqdm(training_instances)):
           
            sampled_data.to(device)

            pred = model(device, sampled_data)

            ground_truth = sampled_data['community', 'interacts_with', 'community'].edge_labels
  
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
        
            total_loss += float(loss.item()) * pred.detach().numel()
            total_examples += pred.detach().numel()
            # if inst_num % 10 == 0: 
            optimizer.step()
            optimizer.zero_grad()
            
        print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
        
        preds = []
        ground_truths = []
        for sampled_data in tqdm(hetero_validation_insts):
            with torch.no_grad():
                sampled_data.to(device)
                preds.append(model(device, sampled_data))
                ground_truths.append(sampled_data['community', 'interacts_with', 'community'].edge_labels)

        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

        # out = torch.sigmoid(torch.cat(preds, dim=0))
        
        # print(set(list(pred.astype(int))), set(list(ground_truth.astype(int))))
        ipp_acc = information_pathway_evaluation(preds, ground_truths)
        auc = roc_auc_score(ground_truth.reshape(len(ground_truth), 1), pred.reshape(len(ground_truth), 1))
        micro_f1 = f1_score(ground_truth.astype(int), pred.astype(int), average="micro")
        macro_f1 = f1_score(ground_truth.astype(int), pred.astype(int), average="macro")
        print(f"Validation AUC: {auc:.4f}")
        print(f"Validation micro f1: {micro_f1:.4f}")
        print(f"Validation macro f1: {macro_f1:.4f}")
        print(f"Validation IPP Accuracy: {ipp_acc:.4f}")
        
        if (auc + micro_f1)/2 > (best_roc_auc + best_micro_f1)/2:
            best_roc_auc = auc
            best_micro_f1 = micro_f1
            
            testing_preds = []
            testing_ground_truths = []
            for sampled_data in tqdm(hetero_testing_insts):
                with torch.no_grad():
                    sampled_data.to(device)
                    testing_preds.append(model(device, sampled_data))
                    testing_ground_truths.append(sampled_data['community', 'interacts_with', 'community'].edge_labels)
            
            testing_pred = torch.cat(testing_preds, dim=0).cpu().numpy()
            testing_ground_truth = torch.cat(testing_ground_truths, dim=0).cpu().numpy()
            
            # print(set(list(testing_pred.astype(int))), set(list(testing_ground_truth.astype(int))))
            testing_ipp_acc = information_pathway_evaluation(testing_preds, testing_ground_truths)
            testing_auc = roc_auc_score(testing_ground_truth.reshape(len(testing_ground_truth), 1), testing_pred.reshape(len(testing_ground_truth), 1))
            testing_micro_f1 = f1_score(testing_ground_truth.astype(int), testing_pred.astype(int), average="micro")
            testing_macro_f1 = f1_score(testing_ground_truth.astype(int), testing_pred.astype(int), average="macro")
            print(f"Testing AUC: {testing_auc:.4f}")
            print(f"Testing micro f1: {testing_micro_f1:.4f}")
            print(f"Testing macro f1: {testing_macro_f1:.4f}")
            print(f"Testing IPP Accuracy: {testing_ipp_acc:.4f}")
            test_results = {
                "Testing AUC": testing_auc,
                "Testing micro f1": testing_micro_f1,
                "Testing macro f1": testing_macro_f1,
                "Testing IPP Accuracy": testing_ipp_acc
            }
            print("saving model...")
            torch.save(model, os.path.join(model_output_path, f"best_{modelname}_noncomm_model.pt"))
            print("saved model!")
            
    print(f"Final Testing Results: {modelname}")
    print("--------------------------")
    for test_result in test_results:
        print(f"{test_result}: {test_results[test_result]}")
    print("--------------------------")
    bl_ground_truths = []
    for sampled_data in tqdm(hetero_testing_insts):
        bl_ground_truths.append(sampled_data['community', 'interacts_with', 'community'].edge_labels)
    
    bl_agg_gts = torch.cat(bl_ground_truths, dim=0).cpu().tolist()
    print("BL Always Negative",sum(bl_agg_gts)/len(bl_agg_gts))
    print("--------------------------")
        
if __name__ == "__main__":
    main()