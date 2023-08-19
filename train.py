import os
import torch
import torch.nn.functional as F
from data_preparation import prepare_training_data
import traceback
from tqdm import tqdm
from model_updated import Model
# from gelato import Gelato
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score, auc, precision_recall_curve, confusion_matrix
import torch.optim as optim
from statistics import mean
import argparse
import utils
import time
import logging

torch.cuda.manual_seed_all(42)

def error_type_analysis(preds, ground_truths):
    cm = confusion_matrix(ground_truths, preds)
    logger.info(f"{cm}")
    tn, fp, fn, tp = cm.ravel()
    return fp/(fp + fn), fn/(fp + fn)

def n_pair_loss(out_pos, out_neg):
    """
    Compute the N-pair loss.

    :param out_pos: similarity scores for positive pairs.
    :param out_neg: similarity scores for negative pairs.
    :return: loss (normalized by the total number of pairs)
    """

    agg_size = out_neg.shape[0] // out_pos.shape[0]  # Number of negative pairs matched to a positive pair.
    agg_size_p1 = agg_size + 1
    agg_size_p1_count = out_neg.shape[0] % out_pos.shape[0]  # Number of positive pairs that should be matched to agg_size + 1 instead because of the remainder.
    out_pos_agg_p1 = out_pos[:agg_size_p1_count].unsqueeze(-1)
    out_pos_agg = out_pos[agg_size_p1_count:].unsqueeze(-1)
    out_neg_agg_p1 = out_neg[:agg_size_p1_count * agg_size_p1].reshape(-1, agg_size_p1)
    out_neg_agg = out_neg[agg_size_p1_count * agg_size_p1:].reshape(-1, agg_size)
    out_diff_agg_p1 = out_neg_agg_p1 - out_pos_agg_p1  # Difference between negative and positive scores.
    out_diff_agg = out_neg_agg - out_pos_agg  # Difference between negative and positive scores.
    out_diff_exp_sum_p1 = torch.exp(torch.clamp(out_diff_agg_p1, max=80.0)).sum(axis=1)
    out_diff_exp_sum = torch.exp(torch.clamp(out_diff_agg, max=80.0)).sum(axis=1)
    out_diff_exp_cat = torch.cat([out_diff_exp_sum_p1, out_diff_exp_sum])
    loss = torch.log(1 + out_diff_exp_cat).sum() / (len(out_pos) + len(out_neg))

    return loss

def get_logger(config):
    timestamp = int(time.time())
    
    commAggPath = os.path.join(config.logger_path, f"{config.community_aggregation_type}")
    commAggPathExist = os.path.exists(commAggPath)
    if not commAggPathExist:
        os.makedirs(commAggPath)
        
    commEmbModelPath = os.path.join(commAggPath, f"{config.community_embedding_model_name}")
    commEmbModelPathExist = os.path.exists(commEmbModelPath)
    if not commEmbModelPathExist:
        os.makedirs(commEmbModelPath)
    
    abl_prefix = config.ablation_type + "_" if config.ablation_type != 'none' else ""
    logger_name = f"{abl_prefix}{timestamp}.log"
    logger_path = os.path.join(commEmbModelPath, logger_name)
    print(f"Logging to {logger_path}")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(logger_path)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
    logger.addHandler(handler)

    return logger

def cross_comm_evaluation(gt_user_user_edges, gt_user_user_labels, pred_comm_to_comm_edges_w_neg, pred_test_comm_comm_labels, comm_path_to_user_path_mapping):
    test_user_user_labels = []
    test_user_user_edges = []
    
    test_eval = {}
    for comm_to_comm_edge, comm_to_comm_label in zip(pred_comm_to_comm_edges_w_neg, pred_test_comm_comm_labels):

        for user_edge in comm_path_to_user_path_mapping[tuple(comm_to_comm_label)]:
            test_user_user_labels.append(comm_to_comm_edge.item())
            test_user_user_edges.append(user_edge)
            # if not tuple(comm_to_comm_label) in test_eval:
            #     test_eval[tuple(comm_to_comm_label)] = {}
            # test_eval[tuple(comm_to_comm_label)][tuple(user_edge)] = (comm_to_comm_edge.item(), gt_user_user_labels[gt_user_user_edges.index(user_edge)])
    
    gt_user_user_edges, gt_user_user_labels = utils.simulsort(gt_user_user_edges, gt_user_user_labels)
    user_level_pathway_with_neg, user_level_pathway_with_neg_labels = utils.simulsort(test_user_user_edges, test_user_user_labels)

    precision, recall, thresholds = precision_recall_curve(gt_user_user_labels, [int(ele) for ele in user_level_pathway_with_neg_labels])
    
    return roc_auc_score(gt_user_user_labels, user_level_pathway_with_neg_labels),\
    f1_score(gt_user_user_labels, [int(ele) for ele in user_level_pathway_with_neg_labels], average="micro"),\
    accuracy_score(gt_user_user_labels, [int(ele) for ele in user_level_pathway_with_neg_labels]),\
    average_precision_score(gt_user_user_labels, [int(ele) for ele in user_level_pathway_with_neg_labels]),\
    auc(recall, precision)


def information_pathway_evaluation(preds, ground_truths):
    agg_acc = []
    for pred,ground_truth in zip(preds, ground_truths):
        agg_acc.append(accuracy_score(ground_truth.cpu().numpy().astype(int), pred.cpu().numpy().astype(int)))

    return mean(agg_acc)

def main(config):    
    
    cached_datapath = os.path.join(config.cached_datapath,f"cached_{config.community_aggregation_type}_{config.community_embedding_model_name}.pkl") 
    training_instances_path = os.path.join(config.info_pathway_instances_datapath, config.training_instances_filename)
    evaluation_instances_path = os.path.join(config.info_pathway_instances_datapath, config.training_instances_filename)
    
    training_community_features_path = os.path.join(config.comm_feats_datapath, config.training_comm_features)
    evaluation_community_features_path = os.path.join(config.comm_feats_datapath, config.eval_comm_features)
    
    training_article_features_path = os.path.join(config.article_features_datapath, config.training_article_features)
    evaluation_article_features_path = os.path.join(config.article_features_datapath, config.eval_article_features)
    
    abl_model_name = config.ablation_type + "_" if config.ablation_type != 'none' else ""
    modelname = f"{abl_model_name}{config.community_aggregation_type}_{config.community_embedding_model_name}_test"

    model_output_path = os.path.join(config.best_model_output_path, f"{modelname}.pt")
     
    training_instances, num_comm_nodes, num_article_nodes, metadata, hetero_validation_insts, hetero_testing_insts, weight_pos= prepare_training_data(training_instances_path, evaluation_instances_path, training_community_features_path, evaluation_community_features_path, training_article_features_path, evaluation_article_features_path, config.subgraph_min_size, config.subgraph_max_size, cached_datapath, overwrite=True)
    
    global logger
    logger = get_logger(config)
    
    out_channels = 42#42 #750#3#1 #32one less
    
    best_micro_f1 = -1
    best_roc_auc = -1
    best_avg_pr = -1
    test_results = {}
    
    logger.info(f"Training Instances: {len(training_instances)}")
    logger.info(f"Evaluation Instances: {len(list(hetero_validation_insts)) +  len(list(hetero_testing_insts))}")
    logger.info(f"Positive Example Weighting: {weight_pos}")
    # model = None
    # if config.auxiliary_model != None:
    #     if config.auxiliary_model == "gelato":
    #         pass
    #     else:
    #         raise NotImplementedError
    # else:
    error_analysis = True
 
    for run in range(0, 3):
        model = Model(hidden_channels=config.hidden_layer_size, out_channels=config.subgraph_max_size, modelname=modelname, num_comm_nodes=num_comm_nodes, num_article_nodes=num_article_nodes, device=device)
    
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=40)
        logger.info(f"-------------------------- Run Number: {str(run)} --------------------------")
        for epoch in range(1, 56):
            logger.info(f"---------------------- Epoch: {str(run)} ----------------------")
            model.train()
            total_loss = total_examples = 0
            for inst_num, sampled_data in enumerate(tqdm(training_instances)):
                sampled_data.to(device)
                
                if config.ablation_type == "none":
                    pred = model(sampled_data)
                else:
                    pred = model.forward_ablate(sampled_data, ablation_type=config.ablation_type)

                ground_truth = sampled_data['community', 'interacts_with', 'community'].edge_labels
        
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth, pos_weight=torch.FloatTensor([weight_pos]).to(device))#sampled_data["wp"].to(device))
                
                loss.backward()
            
                total_loss += float(loss.item()) * pred.detach().numel()
                total_examples += pred.detach().numel()
                # if inst_num % 10 == 0: 
                optimizer.step()
                optimizer.zero_grad()
                
            logger.info(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
            
            preds = []
            ground_truths = []
            
            val_avg_cc_f1 = []
            val_avg_cc_auc = []
            val_avg_cc_acc = []
            val_avg_cc_avg_pr = []
            val_avg_cc_aupr = []
            
            model.eval()
            with torch.no_grad():
                for sampled_data in tqdm(hetero_validation_insts):
                
                    sampled_data.to(device)
                    
                    comm_to_comm_with_neg = sampled_data['community', 'interacts_with', 'community'].edge_labels
                    pred_test_comm_comm_labels = sampled_data['community', 'interacts_with', 'community'].edge_labels_str
                    comm_path_to_user_path_mapping = sampled_data['community', 'interacts_with', 'community'].comm_user_map
                    gt_user_user_labels = sampled_data['community', 'interacts_with', 'community'].user_edge_labels
                    gt_user_user_edges = sampled_data['community', 'interacts_with', 'community'].user_edge_labels_str
                    
                    pred_comm_to_comm_edges_w_neg = model(sampled_data).clamp(min=0,max=1)
                    
                    preds.append(pred_comm_to_comm_edges_w_neg)
                    
                    ground_truths.append(comm_to_comm_with_neg)
                    
                    val_cc_f1, val_cc_auc, val_cc_acc, val_cc_avg_pr, val_cc_aupr = cross_comm_evaluation(gt_user_user_edges, gt_user_user_labels, pred_comm_to_comm_edges_w_neg, pred_test_comm_comm_labels, comm_path_to_user_path_mapping)
                    val_avg_cc_f1.append(val_cc_f1)
                    val_avg_cc_auc.append(val_cc_auc)
                    val_avg_cc_acc.append(val_cc_acc)
                    val_avg_cc_avg_pr.append(val_cc_avg_pr)
                    val_avg_cc_aupr.append(val_cc_aupr)
                    
                pred = torch.cat(preds, dim=0).cpu().numpy()
                ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
                
                ipp_acc = information_pathway_evaluation(preds, ground_truths)
                roc_auc = roc_auc_score(ground_truth.reshape(len(ground_truth), 1), pred.reshape(len(ground_truth), 1))
                micro_f1 = f1_score(ground_truth.astype(int), pred.astype(int), average="micro")
                macro_f1 = f1_score(ground_truth.astype(int), pred.astype(int), average="macro")
                avg_pr = average_precision_score(ground_truth.astype(int), pred.astype(int))
                precision, recall, thresholds = precision_recall_curve(ground_truth.astype(int), pred.astype(int))
                aupr = auc(recall, precision)
                
                logger.info(f"Validation XCOMM AUC: {sum(val_avg_cc_f1)/len(val_avg_cc_f1):.4f}")
                logger.info(f"Validation XCOMM micro f1: {sum(val_avg_cc_auc)/len(val_avg_cc_auc):.4f}")
                logger.info(f"Validation XCOMM IPP Accuracy: {sum(val_avg_cc_acc)/len(val_avg_cc_acc):.4f}")
                logger.info(f"Validation XCOMM Avg. Precision-Recall: {sum(val_avg_cc_avg_pr)/len(val_avg_cc_avg_pr):.4f}")
                logger.info(f"Validation XCOMM AUPR: {sum(val_avg_cc_aupr)/len(val_avg_cc_aupr):.4f}")
                logger.info(f"Validation AUC: {roc_auc:.4f}")
                logger.info(f"Validation micro f1: {micro_f1:.4f}")
                logger.info(f"Validation macro f1: {macro_f1:.4f}")
                logger.info(f"Validation IPP Accuracy: {ipp_acc:.4f}")
                logger.info("--------- Imbalanced Data Metrics ---------")
                logger.info(f"Validation Avg. Precision-Recall: {avg_pr:.4f}")
                logger.info(f"Validation AUPR: {aupr:.4f}")
                error_type_analysis(pred.astype(int), ground_truth.astype(int))
                
                if best_avg_pr < avg_pr:
                    best_avg_pr = avg_pr
                
                    testing_preds = []
                    testing_ground_truths = []
                    
                    test_avg_cc_f1 = []
                    test_avg_cc_auc = []
                    test_avg_cc_acc = []
                    test_avg_cc_avg_pr = []
                    test_avg_cc_aupr = []
                    
                    for sampled_data in tqdm(hetero_testing_insts):
                        
                        sampled_data.to(device)
                        
                        comm_to_comm_with_neg = sampled_data['community', 'interacts_with', 'community'].edge_labels
                        pred_test_comm_comm_labels = sampled_data['community', 'interacts_with', 'community'].edge_labels_str
                        comm_path_to_user_path_mapping = sampled_data['community', 'interacts_with', 'community'].comm_user_map
                        gt_user_user_labels = sampled_data['community', 'interacts_with', 'community'].user_edge_labels
                        gt_user_user_edges = sampled_data['community', 'interacts_with', 'community'].user_edge_labels_str
                    
                        pred_comm_to_comm_edges_w_neg = model(sampled_data).clamp(min=0,max=1)
                        testing_preds.append(pred_comm_to_comm_edges_w_neg)
                        testing_ground_truths.append(comm_to_comm_with_neg)
                        
                        test_cc_f1, test_cc_auc, test_cc_acc, test_cc_avg_pr, test_cc_aupr = cross_comm_evaluation(gt_user_user_edges, gt_user_user_labels, pred_comm_to_comm_edges_w_neg, pred_test_comm_comm_labels, comm_path_to_user_path_mapping)
                        test_avg_cc_f1.append(test_cc_f1)
                        test_avg_cc_auc.append(test_cc_auc)
                        test_avg_cc_acc.append(test_cc_acc)
                        test_avg_cc_avg_pr.append(test_cc_avg_pr)
                        test_avg_cc_aupr.append(test_cc_aupr)
                    
                    testing_pred = torch.cat(testing_preds, dim=0).cpu().numpy()
                    testing_ground_truth = torch.cat(testing_ground_truths, dim=0).cpu().numpy()
                    
                    testing_ipp_acc = information_pathway_evaluation(testing_preds, testing_ground_truths)
                    testing_roc_auc = roc_auc_score(testing_ground_truth.reshape(len(testing_ground_truth), 1), testing_pred.reshape(len(testing_ground_truth), 1))
                    testing_micro_f1 = f1_score(testing_ground_truth.astype(int), testing_pred.astype(int), average="micro")
                    testing_macro_f1 = f1_score(testing_ground_truth.astype(int), testing_pred.astype(int), average="macro")
                    testing_avg_pr = average_precision_score(testing_ground_truth.astype(int), testing_pred.astype(int))
                    testing_precision, testing_recall, testing_thresholds = precision_recall_curve(testing_ground_truth.astype(int), testing_pred.astype(int))
                    testing_aupr = auc(testing_recall, testing_precision)
                    
                    test_results[run] = {
                        "Testing XCOMM AUC": sum(test_avg_cc_f1)/len(test_avg_cc_f1),
                        "Testing XCOMM micro f1": sum(test_avg_cc_auc)/len(test_avg_cc_auc),
                        "Testing XCOMM IPP Accuracy": sum(test_avg_cc_acc)/len(test_avg_cc_acc),
                        "Testing XCOMM Avg Precision-Recall": sum(test_avg_cc_avg_pr)/len(test_avg_cc_avg_pr),
                        "Testing XCOMM AUPR": sum(test_avg_cc_aupr)/len(test_avg_cc_aupr),
                        "Testing AUC": testing_roc_auc,
                        "Testing micro f1": testing_micro_f1,
                        "Testing macro f1": testing_macro_f1,
                        "Testing IPP Accuracy": testing_ipp_acc,
                        "Testing Avg. Precision-Recall": testing_avg_pr,
                        "Testing AUPR": testing_aupr
                    }
                    error_type_analysis(testing_pred.astype(int), testing_ground_truth.astype(int))
                    logger.info("--------------------------")
                    for test_result in test_results[run]:
                        logger.info(f"{test_result}: {test_results[run][test_result]}")
                    logger.info("--------------------------")
                    logger.info("saving model...")
                    torch.save(model, model_output_path)
                    logger.info("saved model!")
            
    logger.info(f"Final Testing Results: {modelname}")
    logger.info("--------------------------")
    copy_paste_string = []
    final_avg_dict = {}
    for run in test_results:
        for test_result in test_results[run]:
            if not test_result in final_avg_dict:
                final_avg_dict[test_result] = []
            final_avg_dict[test_result].append(test_results[run][test_result])
    
    for avg_list in final_avg_dict:
        mean, var = utils.compute_results_stats(final_avg_dict[avg_list])
        logger.info(f"{avg_list}: {mean, var}, {final_avg_dict[avg_list]}")
        copy_paste_string.append(f"{mean},{var},,")
    logger.info("--------------------------")
    logger.info(",".join(copy_paste_string))
    logger.info("--------------------------")
    
    bl_ground_truths = []
    for sampled_data in tqdm(hetero_testing_insts):
        bl_ground_truths.append(sampled_data['community', 'interacts_with', 'community'].edge_labels)
    
    bl_agg_gts = torch.cat(bl_ground_truths, dim=0).cpu().tolist()
    logger.info(f"BL Always Negative: {accuracy_score(bl_agg_gts, [0] * len(bl_agg_gts))}")
    logger.info("--------------------------")
    if config.ablation_type != "none":
        logger.info(f"Ablation Type: {config.ablation_type}")
        
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