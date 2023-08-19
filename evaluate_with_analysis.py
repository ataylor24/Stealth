import os
import torch
import torch.nn.functional as F
from data_preparation import prepare_training_data
import traceback
from tqdm import tqdm
from model_updated import Model
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score, auc, precision_recall_curve, recall_score, confusion_matrix
import torch.optim as optim
from statistics import mean
import argparse
import utils
import plot_utils
from scipy.stats import pearsonr

def generate_plots(proportions, error_analysis_dict, true_pos_analysis_dict):
    def simplify_pos_dict(true_pos_analysis_dict):
        true_pos_count_dict = {}
        total_count_dict = {}
        max_count = 0
        for key in true_pos_analysis_dict:
            max_count = max(max_count, true_pos_analysis_dict[key]["total"])
            true_pos_count_dict[key] = true_pos_analysis_dict[key]["correct"]
            total_count_dict[key] = true_pos_analysis_dict[key]["total"]
        return true_pos_count_dict, total_count_dict, max_count
    
    def simplify_neg_dict(error_analysis_dict):
        error_type_dict = {}
        for key in error_analysis_dict:
            if error_analysis_dict[key]["total"] != 0:
                if error_analysis_dict[key]["false_positive"] + error_analysis_dict[key]["false_negative"] != 0:
                    error_type_dict[key] = (error_analysis_dict[key]["false_positive"] + error_analysis_dict[key]["false_negative"])/error_analysis_dict[key]["total"]
        return error_type_dict
    
    def divide_dict_keys(error_analysis_dict, type_):
        # print(sorted(list(error_analysis_dict.items()), key= lambda k: k[1]["total"]))
        error_type_dict = {}
        for key in error_analysis_dict:
            if error_analysis_dict[key]["total"] != 0:
                if error_analysis_dict[key][type_] != 0:
                    error_type_dict[key] = error_analysis_dict[key][type_]/error_analysis_dict[key]["total"]
        return error_type_dict
    outpath = "/home/ataylor2/processing_covid_tweets/Thunder/error_analysis_plots"
    
    plot_utils.plot_proportion_bar_chart(proportions, xlabel="", ylabel="",title="",outpath=os.path.join(outpath,"proportion_bar_chart.png"))
    
    plot_utils.plot_kde(divide_dict_keys(error_analysis_dict, "false_negative"), xlabel="Ratio of False Negatives to Training Examples", ylabel="",title="",outpath=os.path.join(outpath,"false_neg_error_distribution.png"))
    plot_utils.plot_kde(divide_dict_keys(error_analysis_dict, "false_positive"), xlabel="Ratio of False Positives to Training Examples", ylabel="",title="",outpath=os.path.join(outpath,"false_pos_error_distribution.png"))

    plot_utils.plot_smoothed_line(divide_dict_keys(error_analysis_dict, "false_negative"), xlabel="Ratio of False Negatives to Training Examples", ylabel="",title="",outpath=os.path.join(outpath,"false_neg_error_line.png"))
    plot_utils.plot_smoothed_line(divide_dict_keys(error_analysis_dict, "false_positive"), xlabel="Ratio of False Positives to Training Examples", ylabel="",title="",outpath=os.path.join(outpath,"false_pos_error_line.png"))
    
    true_dict, total_dict, max_count = simplify_pos_dict(true_pos_analysis_dict)
    plot_utils.plot_heatmap(simplify_neg_dict(error_analysis_dict), vmax=max_count, title="",outpath=os.path.join(outpath,"error_heatmap.png"))
    plot_utils.plot_heatmap(true_dict, vmax=max_count, title="",outpath=os.path.join(outpath,"correct_heatmap.png"))
    plot_utils.plot_heatmap(total_dict, vmax=max_count, title="",outpath=os.path.join(outpath,"total_heatmap.png"))


    examine = sorted(list(divide_dict_keys(error_analysis_dict, "false_negative").items()), key= lambda k:k[1])
      
def error_distribution_analysis_train(training_instances):
    error_analysis_dict = {}
    for sampled_data in training_instances:
        comm_to_comm_with_neg = sampled_data['community', 'interacts_with', 'community'].edge_labels
        comm_comm_labels = sampled_data['community', 'interacts_with', 'community'].edge_labels_str
        for gt_edge, comm_to_comm_label in zip(comm_to_comm_with_neg, comm_comm_labels):
            if not tuple(comm_to_comm_label) in error_analysis_dict:
                error_analysis_dict[tuple(comm_to_comm_label)] = 0 #{
                #     # "false_positive": 0,
                #     # "false_negative": 0,
                #     "total": 0
                # }
            error_analysis_dict[tuple(comm_to_comm_label)] += 1#["total"] += 1
            
    return error_analysis_dict

def error_distribution_analysis(pred_comm_to_comm_edges_w_neg, comm_comm_labels, comm_to_comm_with_neg):
    error_analysis_dict = {}
    true_pos_dict = {}
    for pred_edge, gt_edge, comm_to_comm_label in zip(pred_comm_to_comm_edges_w_neg, comm_to_comm_with_neg, comm_comm_labels):
        if not tuple(comm_to_comm_label) in error_analysis_dict:
            error_analysis_dict[tuple(comm_to_comm_label)] = 0 #{
            #     "false_positive": 0,
            #     "false_negative": 0,
            #     "total": 0
            # }
        if not tuple(comm_to_comm_label) in true_pos_dict:
            true_pos_dict[tuple(comm_to_comm_label)] = {
                "correct": 0,
                "total": 0
            }
        # error_analysis_dict[tuple(comm_to_comm_label)]["total"] += 1
        if pred_edge != gt_edge:
            # if gt_edge == 0:
            #     error_analysis_dict[tuple(comm_to_comm_label)]["false_positive"] += 1
            # else:
            #     error_analysis_dict[tuple(comm_to_comm_label)]["false_negative"] += 1
            error_analysis_dict[tuple(comm_to_comm_label)] += 1
    
        else:
            true_pos_dict[tuple(comm_to_comm_label)]["correct"] += 1
        true_pos_dict[tuple(comm_to_comm_label)]["total"] += 1
    return error_analysis_dict, true_pos_dict

def error_type_analysis(preds, ground_truths):
    cm = confusion_matrix(ground_truths, preds)
    tn, fp, fn, tp = cm.ravel()

    return fp, fn

def cross_comm_err_evaluation(gt_user_user_edges, gt_user_user_labels, pred_comm_to_comm_edges_w_neg, pred_test_comm_comm_labels, comm_path_to_user_path_mapping):
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
    # print(list(zip(pred_comm_to_comm_edges_w_neg, pred_test_comm_comm_labels)))
    # print(list(zip(user_level_pathway_with_neg, user_level_pathway_with_neg_labels)))
   
    
    return error_type_analysis(gt_user_user_labels, [int(ele) for ele in user_level_pathway_with_neg_labels])
    


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
    # print(list(zip(pred_comm_to_comm_edges_w_neg, pred_test_comm_comm_labels)))
    # print(list(zip(user_level_pathway_with_neg, user_level_pathway_with_neg_labels)))
   
    
    return roc_auc_score(gt_user_user_labels, user_level_pathway_with_neg_labels),\
    f1_score(gt_user_user_labels, [int(ele) for ele in user_level_pathway_with_neg_labels], average="micro"),\
    accuracy_score(gt_user_user_labels, [int(ele) for ele in user_level_pathway_with_neg_labels])


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
    
    modelname = f"{config.community_aggregation_type}_{config.community_embedding_model_name}"
    
    model_output_path = os.path.join(config.best_model_output_path, f"{modelname}_test.pt")

    training_instances, num_comm_nodes, num_article_nodes, metadata, hetero_validation_insts, hetero_testing_insts, _ = prepare_training_data(training_instances_path, evaluation_instances_path, training_community_features_path, evaluation_community_features_path, training_article_features_path, evaluation_article_features_path, config.subgraph_min_size, config.subgraph_max_size, cached_datapath, overwrite=False)
        
    # subgraph_max_size = 25 #500
    # subgraph_min_size = 4

    test_results = {}
    
    print("Validation instances", len(hetero_validation_insts))
    print("Testing instances", len(hetero_testing_insts))
    print(model_output_path)
    model = utils.load_model(model_output_path)
    model = model.to(device)
  
    preds = []
    ground_truths = []
    
    val_avg_cc_f1 = []
    val_avg_cc_auc = []
    val_avg_cc_acc = []
    
    error_analysis_eval_dict = {}
    error_analysis_dict = {}
    true_pos_analysis_dict = {}
    total_cc_fp_ = 0
    total_cc_fn_ = 0
    
    model.eval()
    with torch.no_grad():
        # for sampled_data in tqdm(hetero_validation_insts):
        
        #     sampled_data.to(device)
            
        #     comm_to_comm_with_neg = sampled_data['community', 'interacts_with', 'community'].edge_labels
        #     pred_test_comm_comm_labels = sampled_data['community', 'interacts_with', 'community'].edge_labels_str
        #     comm_path_to_user_path_mapping = sampled_data['community', 'interacts_with', 'community'].comm_user_map
        #     gt_user_user_labels = sampled_data['community', 'interacts_with', 'community'].user_edge_labels
        #     gt_user_user_edges = sampled_data['community', 'interacts_with', 'community'].user_edge_labels_str
            
        #     pred_comm_to_comm_edges_w_neg = model(sampled_data).clamp(min=0,max=1)
            
        #     preds.append(pred_comm_to_comm_edges_w_neg)
            
        #     ground_truths.append(comm_to_comm_with_neg)
            
        #     val_cc_f1, val_cc_auc, val_cc_acc = cross_comm_evaluation(gt_user_user_edges, gt_user_user_labels, pred_comm_to_comm_edges_w_neg, pred_test_comm_comm_labels, comm_path_to_user_path_mapping)
        #     val_avg_cc_f1.append(val_cc_f1)
        #     val_avg_cc_auc.append(val_cc_auc)
        #     val_avg_cc_acc.append(val_cc_acc)
            
        # pred = torch.cat(preds, dim=0).cpu().numpy()
        # ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

        # ipp_acc = information_pathway_evaluation(preds, ground_truths)
        # roc_auc = roc_auc_score(ground_truth.reshape(len(ground_truth), 1), pred.reshape(len(ground_truth), 1))
        # micro_f1 = f1_score(ground_truth.astype(int), pred.astype(int), average="micro")
        # macro_f1 = f1_score(ground_truth.astype(int), pred.astype(int), average="macro")
        # avg_pr = average_precision_score(ground_truth.astype(int), pred.astype(int))
        # precision, recall, thresholds = precision_recall_curve(ground_truth.astype(int), pred.astype(int))
        # aupr = auc(recall, precision)
        
        # print(f"Validation XCOMM AUC: {sum(val_avg_cc_f1)/len(val_avg_cc_f1):.4f}")
        # print(f"Validation XCOMM micro f1: {sum(val_avg_cc_auc)/len(val_avg_cc_auc):.4f}")
        # print(f"Validation XCOMM IPP Accuracy: {sum(val_avg_cc_acc)/len(val_avg_cc_acc):.4f}")
        
        # print(f"Validation AUC: {roc_auc:.4f}")
        # print(f"Validation micro f1: {micro_f1:.4f}")
        # print(f"Validation macro f1: {macro_f1:.4f}")
        # print(f"Validation IPP Accuracy: {ipp_acc:.4f}")
        # print("--------- Imbalanced Data Metrics ---------")
        # print(f"Validation Avg. Precision-Recall: {avg_pr:.4f}")
        # print(f"Validation AUPR: {aupr:.4f}")
        
        # best_roc_auc = (sum(val_avg_cc_f1)/len(val_avg_cc_f1))
        
        testing_preds = []
        testing_ground_truths = []
        
        test_avg_cc_f1 = []
        test_avg_cc_auc = []
        test_avg_cc_acc = []
        for sampled_data in tqdm(hetero_testing_insts):
            sampled_data.to(device)
            
            comm_to_comm_with_neg = sampled_data['community', 'interacts_with', 'community'].edge_labels
            pred_test_comm_comm_labels = sampled_data['community', 'interacts_with', 'community'].edge_labels_str
            comm_path_to_user_path_mapping = sampled_data['community', 'interacts_with', 'community'].comm_user_map
            gt_user_user_labels = sampled_data['community', 'interacts_with', 'community'].user_edge_labels
            gt_user_user_edges = sampled_data['community', 'interacts_with', 'community'].user_edge_labels_str
        
            # pred_comm_to_comm_edges_w_neg = model(data=sampled_data,device=device).clamp(min=0,max=1)
            pred_comm_to_comm_edges_w_neg = model(sampled_data).clamp(min=0,max=1)
            testing_preds.append(pred_comm_to_comm_edges_w_neg)
            testing_ground_truths.append(comm_to_comm_with_neg)
            
            test_cc_f1, test_cc_auc, test_cc_acc = cross_comm_evaluation(gt_user_user_edges, gt_user_user_labels, pred_comm_to_comm_edges_w_neg, pred_test_comm_comm_labels, comm_path_to_user_path_mapping)
            test_avg_cc_f1.append(test_cc_f1)
            test_avg_cc_auc.append(test_cc_auc)
            test_avg_cc_acc.append(test_cc_acc)
            
            cc_fp, cc_fn = cross_comm_err_evaluation(gt_user_user_edges, gt_user_user_labels, pred_comm_to_comm_edges_w_neg, pred_test_comm_comm_labels, comm_path_to_user_path_mapping)
            total_cc_fp_ += cc_fp
            total_cc_fn_ += cc_fn
            # print(cc_fp, cc_fn)
            # print(total_cc_fp, total_cc_fn)
            # print("-----------------------")
            
            sub_error_analysis_dict, sub_true_pos_analysis_dict = error_distribution_analysis(pred_comm_to_comm_edges_w_neg, pred_test_comm_comm_labels, comm_to_comm_with_neg)
            utils.merge_dicts_add(error_analysis_eval_dict, sub_error_analysis_dict)
            utils.merge_dicts_add(true_pos_analysis_dict, sub_true_pos_analysis_dict)
        testing_pred = torch.cat(testing_preds, dim=0).cpu().numpy()
        testing_ground_truth = torch.cat(testing_ground_truths, dim=0).cpu().numpy()
        
        utils.merge_dicts_add(error_analysis_dict,error_distribution_analysis_train(training_instances))
        
        false_positive_rate, false_negative_rate = error_type_analysis(testing_pred.astype(int), testing_ground_truth.astype(int))
        proportions = {
            "False Positive Rate": false_positive_rate,
            "False Negative Rate": false_negative_rate
        }
        
        # print(set(list(testing_pred.astype(int))), set(list(testing_ground_truth.astype(int))))
        testing_ipp_acc = information_pathway_evaluation(testing_preds, testing_ground_truths)
        testing_roc_auc = roc_auc_score(testing_ground_truth.reshape(len(testing_ground_truth), 1), testing_pred.reshape(len(testing_ground_truth), 1))
        testing_micro_f1 = f1_score(testing_ground_truth.astype(int), testing_pred.astype(int), average="micro")
        testing_macro_f1 = f1_score(testing_ground_truth.astype(int), testing_pred.astype(int), average="macro")
        testing_avg_pr = average_precision_score(testing_ground_truth.astype(int), testing_pred.astype(int))
        testing_precision, testing_recall, testing_thresholds = precision_recall_curve(testing_ground_truth.astype(int), testing_pred.astype(int))
        testing_aupr = auc(testing_recall, testing_precision)
        
        test_results = {
            "Testing XCOMM AUC": sum(test_avg_cc_f1)/len(test_avg_cc_f1),
            "Testing XCOMM micro f1": sum(test_avg_cc_auc)/len(test_avg_cc_auc),
            "Testing XCOMM IPP Accuracy": sum(test_avg_cc_acc)/len(test_avg_cc_acc),
            "Testing AUC": testing_roc_auc,
            "Testing micro f1": testing_micro_f1,
            "Testing macro f1": testing_macro_f1,
            "Testing IPP Accuracy": testing_ipp_acc,
            "Testing Avg. Precision-Recall": testing_avg_pr,
            "Testing AUPR": testing_aupr
        }

            
    print(f"Final Testing Results: {modelname}")
    print("--------------------------")
    copy_paste_string = []
    for test_result in test_results:
        print(f"{test_result}: {test_results[test_result]}")
        copy_paste_string.append(str(test_results[test_result]))
    print("--------------------------")
    print(",".join(copy_paste_string))
    print("--------------------------")
    
    bl_ground_truths = []
    for sampled_data in tqdm(hetero_testing_insts):
        bl_ground_truths.append(sampled_data['community', 'interacts_with', 'community'].edge_labels)
    
    bl_agg_gts = torch.cat(bl_ground_truths, dim=0).cpu().tolist()
    # print("BL Always Negative", accuracy_score(bl_agg_gts, [0] * len(bl_agg_gts)))
    # print("--------------------------")
    print(modelname, [v for k,v in proportions.items()])
    print("xcomm error rates", total_cc_fp_/(total_cc_fp_ + total_cc_fn_), total_cc_fn_/(total_cc_fp_ + total_cc_fn_))
    # generate_plots(proportions, error_analysis_dict, true_pos_analysis_dict)
    # print(error_analysis_eval_dict)
    # print(error_analysis_dict)
    out_dict = {}
    total_train = []
    total_incorrect = []
    for eval_pair_key in error_analysis_eval_dict:
        if eval_pair_key in error_analysis_dict:
            out_dict[eval_pair_key] = (error_analysis_eval_dict[eval_pair_key], error_analysis_dict[eval_pair_key])
            # print(f"{error_analysis_dict[eval_pair_key]},{eval_pair_key[0]},{eval_pair_key[1]},{error_analysis_eval_dict[eval_pair_key]}")
            total_train.append(error_analysis_dict[eval_pair_key])
            total_incorrect.append(error_analysis_eval_dict[eval_pair_key])

    print("pearson", pearsonr(total_train, total_incorrect)[0])
    
    
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