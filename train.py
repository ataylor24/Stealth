import os
import torch
import torch.nn.functional as F
from data_preparation import prepare_training_data
import traceback
from tqdm import tqdm
from model import Model
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import torch.optim as optim
from statistics import mean
import argparse
import utils

def cross_comm_evaluation(gt_user_user_edges, gt_user_user_labels, pred_comm_to_comm_edges_w_neg, pred_test_comm_comm_labels, comm_path_to_user_path_mapping):
    test_user_user_labels = []
    test_user_user_edges = []
    for comm_to_comm_edge, comm_to_comm_label in zip(pred_comm_to_comm_edges_w_neg, pred_test_comm_comm_labels):

        for user_edge in comm_path_to_user_path_mapping[tuple(comm_to_comm_label)]:
            test_user_user_labels.append(comm_to_comm_edge.item())
            test_user_user_edges.append(user_edge)
    
    gt_user_user_edges, gt_user_user_labels = utils.simulsort(gt_user_user_edges, gt_user_user_labels)
    user_level_pathway_with_neg, user_level_pathway_with_neg_labels = utils.simulsort(test_user_user_edges, test_user_user_labels)

    return roc_auc_score(gt_user_user_labels, user_level_pathway_with_neg_labels),\
    f1_score(gt_user_user_labels, [int(ele) for ele in user_level_pathway_with_neg_labels], average="micro"),\
    accuracy_score(gt_user_user_labels, [int(ele) for ele in user_level_pathway_with_neg_labels])


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

def main(config):
    
    cached_datapath = os.path.join(config.cached_datapath,f"cached_{config.community_aggregation_type}_{config.community_embedding_model_name}.pkl") 
    training_instances_path = os.path.join(config.info_pathway_instances_datapath, config.training_instances_filename)
    evaluation_instances_path = os.path.join(config.info_pathway_instances_datapath, config.training_instances_filename)
    
    training_community_features_path = os.path.join(config.comm_feats_datapath, config.training_comm_features)
    evaluation_community_features_path = os.path.join(config.comm_feats_datapath, config.eval_comm_features)
    
    training_article_features_path = os.path.join(config.article_features_datapath, config.training_article_features)
    evaluation_article_features_path = os.path.join(config.article_features_datapath, config.eval_article_features)
    
    model_output_path = os.path.join(config.best_model_output_path, f"{config.community_aggregation_type}_{config.community_embedding_model_name}.pt")

    training_instances, num_comm_nodes, num_article_nodes, metadata, hetero_validation_insts, hetero_testing_insts = prepare_training_data(training_instances_path, evaluation_instances_path, training_community_features_path, evaluation_community_features_path, training_article_features_path, evaluation_article_features_path, config.subgraph_min_size, config.subgraph_max_size, cached_datapath, overwrite=True)
        
    # subgraph_max_size = 25 #500
    # subgraph_min_size = 4
 
    modelname = "gat" #gat or graphsage
    out_channels = 42#750#3#1 #32one less
    
    best_micro_f1 = -1
    best_roc_auc = -1
    test_results = {}
    
    # training_instances, num_comm_nodes, num_article_nodes, metadata, \
    #     hetero_validation_insts, hetero_testing_insts = prepare_training_data(training_g_instances_datapath, \
    #         evaluation_g_instances_datapath, training_comm_mapping_datapath, training_comm_feats_datapath, \
    #     eval_comm_feats_datapath, training_article_feats_datapath, eval_article_feats_datapath, subgraph_min_size, subgraph_max_size, tt_split, data_subset, cached_datapath)    
    
    print("--------------------------")
    bl_ground_truths = []
    for sampled_data in tqdm(hetero_testing_insts):
        # print(sampled_data['community', 'interacts_with', 'community'])
        bl_ground_truths.append(sampled_data['community', 'interacts_with', 'community'].edge_labels)
    
    bl_agg_gts = torch.cat(bl_ground_truths, dim=0).cpu().tolist()
    print("BL Random",sum(bl_agg_gts)/len(bl_agg_gts))
    print("--------------------------")
    
    print("training_instances", len(training_instances))
    model = Model(hidden_channels=config.subgraph_max_size, out_channels=out_channels, modelname=modelname, num_comm_nodes=num_comm_nodes, num_article_nodes=num_article_nodes, metadata=metadata, embedding_dim=config.embedding_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=40)
 
    
    model.train()
    for epoch in range(1, 50):
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
        
        val_avg_cc_f1 = []
        val_avg_cc_auc = []
        val_avg_cc_acc = []
        
        for sampled_data in tqdm(hetero_validation_insts):
            with torch.no_grad():
                sampled_data.to(device)
                
                comm_to_comm_with_neg = sampled_data['community', 'interacts_with', 'community'].edge_labels
                pred_test_comm_comm_labels = sampled_data['community', 'interacts_with', 'community'].edge_labels_str
                comm_path_to_user_path_mapping = sampled_data['community', 'interacts_with', 'community'].comm_user_map
                gt_user_user_labels = sampled_data['community', 'interacts_with', 'community'].user_edge_labels
                gt_user_user_edges = sampled_data['community', 'interacts_with', 'community'].user_edge_labels_str
                
                pred_comm_to_comm_edges_w_neg = model(device, sampled_data)
                
                preds.append(pred_comm_to_comm_edges_w_neg)
                
                ground_truths.append(comm_to_comm_with_neg)
                
                val_cc_f1, val_cc_auc, val_cc_acc = cross_comm_evaluation(gt_user_user_edges, gt_user_user_labels, pred_comm_to_comm_edges_w_neg, pred_test_comm_comm_labels, comm_path_to_user_path_mapping)
                val_avg_cc_f1.append(val_cc_f1)
                val_avg_cc_auc.append(val_cc_auc)
                val_avg_cc_acc.append(val_cc_acc)
                
        pred = torch.cat(preds, dim=0).cpu().numpy()
        ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

        ipp_acc = information_pathway_evaluation(preds, ground_truths)
        auc = roc_auc_score(ground_truth.reshape(len(ground_truth), 1), pred.reshape(len(ground_truth), 1))
        micro_f1 = f1_score(ground_truth.astype(int), pred.astype(int), average="micro")
        macro_f1 = f1_score(ground_truth.astype(int), pred.astype(int), average="macro")
        
        print(f"Validation XCOMM AUC: {sum(val_avg_cc_f1)/len(val_avg_cc_f1):.4f}")
        print(f"Validation XCOMM micro f1: {sum(val_avg_cc_auc)/len(val_avg_cc_auc):.4f}")
        print(f"Validation XCOMM IPP Accuracy: {sum(val_avg_cc_acc)/len(val_avg_cc_acc):.4f}")
        
        print(f"Validation AUC: {auc:.4f}")
        print(f"Validation micro f1: {micro_f1:.4f}")
        print(f"Validation macro f1: {macro_f1:.4f}")
        print(f"Validation IPP Accuracy: {ipp_acc:.4f}")
        
        if (sum(val_avg_cc_f1)/len(val_avg_cc_f1)) > best_roc_auc:
            best_roc_auc = (sum(val_avg_cc_f1)/len(val_avg_cc_f1))
           
            testing_preds = []
            testing_ground_truths = []
            
            test_avg_cc_f1 = []
            test_avg_cc_auc = []
            test_avg_cc_acc = []
            for sampled_data in tqdm(hetero_testing_insts):
                with torch.no_grad():
                    sampled_data.to(device)
                    
                    comm_to_comm_with_neg = sampled_data['community', 'interacts_with', 'community'].edge_labels
                    pred_test_comm_comm_labels = sampled_data['community', 'interacts_with', 'community'].edge_labels_str
                    comm_path_to_user_path_mapping = sampled_data['community', 'interacts_with', 'community'].comm_user_map
                    gt_user_user_labels = sampled_data['community', 'interacts_with', 'community'].user_edge_labels
                    gt_user_user_edges = sampled_data['community', 'interacts_with', 'community'].user_edge_labels_str
                
                    pred_comm_to_comm_edges_w_neg = model(device, sampled_data)
                    testing_preds.append(pred_comm_to_comm_edges_w_neg)
                    testing_ground_truths.append(comm_to_comm_with_neg)
                    
                    test_cc_f1, test_cc_auc, test_cc_acc = cross_comm_evaluation(gt_user_user_edges, gt_user_user_labels, pred_comm_to_comm_edges_w_neg, pred_test_comm_comm_labels, comm_path_to_user_path_mapping)
                    test_avg_cc_f1.append(test_cc_f1)
                    test_avg_cc_auc.append(test_cc_auc)
                    test_avg_cc_acc.append(test_cc_acc)
            
            testing_pred = torch.cat(testing_preds, dim=0).cpu().numpy()
            testing_ground_truth = torch.cat(testing_ground_truths, dim=0).cpu().numpy()
            
            # print(set(list(testing_pred.astype(int))), set(list(testing_ground_truth.astype(int))))
            testing_ipp_acc = information_pathway_evaluation(testing_preds, testing_ground_truths)
            testing_auc = roc_auc_score(testing_ground_truth.reshape(len(testing_ground_truth), 1), testing_pred.reshape(len(testing_ground_truth), 1))
            testing_micro_f1 = f1_score(testing_ground_truth.astype(int), testing_pred.astype(int), average="micro")
            testing_macro_f1 = f1_score(testing_ground_truth.astype(int), testing_pred.astype(int), average="macro")
            
            print(f"Testing XCOMM AUC: {sum(test_avg_cc_f1)/len(test_avg_cc_f1):.4f}")
            print(f"Testing XCOMM micro f1: {sum(test_avg_cc_auc)/len(test_avg_cc_auc):.4f}")
            print(f"Testing XCOMM IPP Accuracy: {sum(test_avg_cc_acc)/len(test_avg_cc_acc):.4f}")
        
            print(f"Testing AUC: {testing_auc:.4f}")
            print(f"Testing micro f1: {testing_micro_f1:.4f}")
            print(f"Testing macro f1: {testing_macro_f1:.4f}")
            print(f"Testing IPP Accuracy: {testing_ipp_acc:.4f}")
            test_results = {
                "Testing XCOMM AUC": sum(test_avg_cc_f1)/len(test_avg_cc_f1),
                "Testing XCOMM micro f1": sum(test_avg_cc_auc)/len(test_avg_cc_auc),
                "Testing XCOMM IPP Accuracy": sum(test_avg_cc_acc)/len(test_avg_cc_acc),
                "Testing AUC": testing_auc,
                "Testing micro f1": testing_micro_f1,
                "Testing macro f1": testing_macro_f1,
                "Testing IPP Accuracy": testing_ipp_acc
            }
            print("saving model...")
            torch.save(model, model_output_path)
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
    parser = argparse.ArgumentParser(description='Read configuration JSON file')
    parser.add_argument('config', type=str, help='Path to the configuration JSON file')
    args = parser.parse_args()

    config = utils.read_config(args.config)
    config.update(args.__dict__)
    config = argparse.Namespace(**config)
    global device
    device = config.cuda_device
    main(config)