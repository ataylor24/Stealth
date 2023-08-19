import os, sys, mmap
from tqdm.auto import tqdm
import dill
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import utils

def construct_and_map_information_pathways(tweet_list_file_path, community_assignments_by_comm, debug=False):
    community_assignments_by_user = utils.reverse_two_layer_dict(community_assignments_by_comm)

    comm_mapping = utils.create_key_mapping(community_assignments_by_comm)
    
    tweet_tree_list = utils.load_dill(tweet_list_file_path)
    information_pathways = {}
    for tweet_tree in tqdm(tweet_tree_list):
        information_pathway = {
            'tweet_ids': {}, 
            'user_to_user': set(),
            'comm_to_comm': set(),
            'comm_to_comm_with_neg': set(),
            'user_level_pathway_with_neg': set(),
            'comm_to_comm_with_neg_labels': set(),
            'user_level_pathway_with_neg_labels': set(), 
            "comm_path_to_user_path_mapping": {},
            'article_to_comm': None, 
            'article_from_news_source': None
        }
        
        user_neighborhood_dict = {}
        source_set = set()
        skip_tweet = False
        
        information_pathway['tweet_ids'] = tweet_tree_list[tweet_tree]['tweet_ids']
        
        if debug:
            test_user_edges = []
            test_mapped_user_edges = {}
               
        for (src_user, trg_user) in tweet_tree_list[tweet_tree]['comm_to_comm']:
            try:
                source_comm = community_assignments_by_user[src_user]
                target_comm = community_assignments_by_user[trg_user]
                
                information_pathway['user_to_user'].add((src_user, trg_user))
                source_set.add(src_user)
                source_set.add(trg_user)
                information_pathway['comm_to_comm'].add((source_comm, target_comm))
                if debug:
                    test_user_edges.append((src_user, trg_user))
                    
                target_comm = community_assignments_by_user[trg_user]
                user_neighborhood_dict[src_user] = {}
                
                if not target_comm in user_neighborhood_dict[src_user]:
                    user_neighborhood_dict[src_user][target_comm] = set()
                user_neighborhood_dict[src_user][target_comm].add(trg_user)
                
                if not (source_comm, target_comm) in information_pathway['comm_path_to_user_path_mapping']:
                    information_pathway['comm_path_to_user_path_mapping'][(source_comm, target_comm)] = []
                information_pathway['comm_path_to_user_path_mapping'][(source_comm, target_comm)].append((src_user, trg_user))
                if debug:
                    # test_mapped_user_edges.append((source_user, target_user))
                    if not (source_comm, target_comm) in test_mapped_user_edges:
                        test_mapped_user_edges[(source_comm, target_comm)] = []
                    test_mapped_user_edges[(source_comm, target_comm)].append((src_user, trg_user))
            except KeyError:
                skip_tweet = True
        if skip_tweet: continue
       
        if len(information_pathway['comm_to_comm']) < 3:
            continue
        
        try:
            information_pathway['article_to_comm'] = [tweet_tree_list[tweet_tree]['article_to_comm'][0], community_assignments_by_user[tweet_tree_list[tweet_tree]['article_to_comm'][1]]]
        except:
            continue
        try:
            information_pathway['article_from_news_source'] = tweet_tree_list[tweet_tree]['article_from_news_source']
        except:
            continue
        
        if len(tweet_tree_list[tweet_tree]['comm_to_comm']) > 42:
            continue
        
        avg_info_pathway_size.append(len(information_pathway["comm_to_comm"]))
        
        num_neg_edges = 42 - len(information_pathway["comm_to_comm"])
        
        source_distribution = list(source_set) 
        target_distribution = list(community_assignments_by_user.items())
        
        sampled_negative_uu_edges = []
        sampled_negative_cc_edges = set()
        valid_negative_sample = False
            
        while len(sampled_negative_cc_edges) < num_neg_edges:
            source_user = random.choice(source_distribution)
            source_community = community_assignments_by_user[source_user]
            target_user, target_community = random.choice(target_distribution)
            if source_user == target_user or source_community == target_community:
                continue
            if not target_community in user_neighborhood_dict and not (source_community, target_community) in information_pathway['comm_to_comm']: #or not target_user in user_neighborhood_dict[target_community]:
                
                sampled_negative_uu_edges.append((source_user, target_user))
                if debug:
                    test_user_edges.append((source_user, target_user))
                sampled_negative_cc_edges.add((source_community, target_community))
                
                if not (source_community, target_community) in information_pathway['comm_path_to_user_path_mapping']:
                    information_pathway['comm_path_to_user_path_mapping'][(source_community, target_community)] = []
                information_pathway['comm_path_to_user_path_mapping'][(source_community, target_community)].append((source_user, target_user))
                if debug:
                    # test_mapped_user_edges.append((source_user, target_user))
                    if not (source_community, target_community) in test_mapped_user_edges:
                        test_mapped_user_edges[(source_community, target_community)] = []
                    test_mapped_user_edges[(source_community, target_community)].append((source_user, target_user))
                
        
        sampled_negative_uu_edges = list(sampled_negative_uu_edges)
        sampled_negative_cc_edges = list(sampled_negative_cc_edges)
        sampled_negative_uu_labels = [0] * len(sampled_negative_uu_edges)
        sampled_negative_cc_labels = [0] * len(sampled_negative_cc_edges)
                
        user_to_user_edges = list(information_pathway['user_to_user'])
        comm_to_comm_edges = list(information_pathway["comm_to_comm"])
        positive_uu_labels = [1] * len(user_to_user_edges)
        positive_cc_labels = [1] * len(comm_to_comm_edges)
        
        user_to_user_edges_w_neg = user_to_user_edges + sampled_negative_uu_edges
        user_to_user_labels_w_neg = positive_uu_labels + sampled_negative_uu_labels

        comm_to_comm_edges_w_neg = comm_to_comm_edges + sampled_negative_cc_edges
        comm_to_comm_labels_w_neg = positive_cc_labels + sampled_negative_cc_labels

        shuffled_comm_to_comm_edges_w_neg, shuffled_comm_to_comm_labels_w_neg = utils.two_list_shuffle(comm_to_comm_edges_w_neg, comm_to_comm_labels_w_neg)
        
        information_pathway['comm_to_comm_with_neg'] = shuffled_comm_to_comm_edges_w_neg
        information_pathway['comm_to_comm_with_neg_labels'] = shuffled_comm_to_comm_labels_w_neg
        information_pathway['user_level_pathway_with_neg'] = user_to_user_edges_w_neg 
        information_pathway['user_level_pathway_with_neg_labels'] = user_to_user_labels_w_neg
        
        if debug:
            if not (sorted(shuffled_comm_to_comm_edges_w_neg) == sorted(comm_to_comm_edges_w_neg)\
            and sorted(shuffled_comm_to_comm_labels_w_neg) == sorted(comm_to_comm_labels_w_neg)):
                print("FIX ISSUE WITH SHUFFLING")
            
            recon_user_level_pathway = []
            for edge in information_pathway['comm_to_comm_with_neg']:
                recon_user_level_pathway.extend(information_pathway['comm_path_to_user_path_mapping'][edge])
            if sorted(recon_user_level_pathway) != sorted(information_pathway['user_level_pathway_with_neg']):
                print(recon_user_level_pathway)
                print(information_pathway['user_level_pathway_with_neg'])
                overlap = 0
                for ele in recon_user_level_pathway:
                    if ele in information_pathway['user_level_pathway_with_neg']:
                        overlap += 1
                print(f"overlap: {overlap}/{len(information_pathway['user_level_pathway_with_neg'])}, len(recon_user_level_pathway): {len(recon_user_level_pathway)}")
                print(set(information_pathway['user_level_pathway_with_neg']).difference(set(recon_user_level_pathway)))
                print(len(information_pathway['user_level_pathway_with_neg']) == len(recon_user_level_pathway))
                print("FIX ISSUE WITH MAPPING")
                
                if not (source_community, target_community) in test_mapped_user_edges:
                    test_mapped_user_edges[(source_community, target_community)] = []
                test_mapped_user_edges_mapped = [(source_user, target_user) for (source_community, target_community) in test_mapped_user_edges for (source_user, target_user) in test_mapped_user_edges[(source_community, target_community)]]
                print(f"Mapping ISSUE: {sorted(test_user_edges) != sorted(test_mapped_user_edges_mapped)}")
                print(len(test_mapped_user_edges_mapped), len(test_user_edges))
                if test_user_edges != test_mapped_user_edges_mapped:
                    for element in test_user_edges:
                        if not element in test_mapped_user_edges_mapped:
                            print("HERE")
                    for element in test_mapped_user_edges_mapped:
                        if not element in test_user_edges:
                            print("HERE")
                
            
         
            #testing eval
            # zero_neg_count = 0
            # neg_count = random.randint(1, int(len(information_pathway['comm_to_comm_with_neg_labels'])/2))
            # pos_count = len(information_pathway['comm_to_comm_with_neg_labels']) - zero_neg_count

            # unshuffled_test_comm_comm_labels = [0]*zero_neg_count + [1]*pos_count
            # random.shuffle(unshuffled_test_comm_comm_labels)
            
            # shuffled_test_comm_comm_labels = unshuffled_test_comm_comm_labels
            # from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
            
            # test_user_user_labels = []
            # test_user_user_edges = []
            # for comm_to_comm_edge, comm_to_comm_label in zip(shuffled_comm_to_comm_edges_w_neg, shuffled_test_comm_comm_labels):
            #     for user_edge in information_pathway['comm_path_to_user_path_mapping'][comm_to_comm_edge]:
            #         test_user_user_labels.append(comm_to_comm_label)
            #         test_user_user_edges.append(user_edge)
            
            # test_user_user_edges, test_user_user_labels = utils.simulsort(test_user_user_edges, test_user_user_labels)
            # user_level_pathway_with_neg, user_level_pathway_with_neg_labels = utils.simulsort(information_pathway['user_level_pathway_with_neg'], information_pathway['user_level_pathway_with_neg_labels'])
            
            # if test_user_user_edges != user_level_pathway_with_neg:
            #     print("FIX ISSUE WITH UNSORTED MAPPING")
            # else:
            #     print(f"roc_auc_score: {roc_auc_score(user_level_pathway_with_neg_labels,test_user_user_labels)}")
            #     print(f"f1_score: {f1_score(user_level_pathway_with_neg_labels,test_user_user_labels)}")
            #     print(f"accuracy_score: {accuracy_score(user_level_pathway_with_neg_labels,test_user_user_labels)}")
                
        information_pathways[tweet_tree] = information_pathway
      
    return information_pathways
            
def retrieve_tree_files(dir_path):
    files = []
    for filename in os.listdir(dir_path):
        if not "_instances" in filename:
            continue
        files.append(os.path.join(dir_path, filename))
    return files

    
def main():
    global avg_info_pathway_size
    avg_info_pathway_size = []
    
    utils.set_news_urls()
    
    tweet_tree_path = "/home/ataylor2/processing_covid_tweets/Thunder/full_tweet_trees"
    
    #eng, int, ips
    community_types = ["srd","trd"]#["eng", "int", "ips"]
    
    tree_files = retrieve_tree_files(tweet_tree_path)
    community_type_information_pathways = {}
    for i, window_file in enumerate(tree_files):
        window_name = f"window_{i+1}"
        community_type_information_pathways[window_name] = {}
        for community_type in community_types:
            if community_type == "eng":
                # Engagement
                comm_fp = "/home/ataylor2/processing_covid_tweets/Thunder/community_assignments/engagement_comms.json"
            elif community_type == "int":
                # Interaction
                comm_fp = "/home/ataylor2/processing_covid_tweets/Thunder/community_assignments/interaction_comms.json"
            elif community_type == "ips":
                # Influence and Passivity
                comm_fp = "/home/ataylor2/processing_covid_tweets/Thunder/community_assignments/inf_pass_comms.json" 
            elif community_type == "srd":
                # Semi Random Communities
                comm_fp = "/home/ataylor2/processing_covid_tweets/Thunder/community_assignments/semi_random_comms.json"            
            elif community_type == "trd":
                # True Random Communities
                comm_fp = "/home/ataylor2/processing_covid_tweets/Thunder/community_assignments/true_random_comms.json"
            print(community_type)
            community_type_information_pathways[window_name][community_type] = construct_and_map_information_pathways(window_file, utils.load_json(comm_fp))
    
    filtered_community_type_information_pathways = {}
    
    for window in community_type_information_pathways:
        info_pathway_ids_intersection = None
        for community_type in community_type_information_pathways[window]:
            if info_pathway_ids_intersection == None:
                info_pathway_ids_intersection = set(community_type_information_pathways[window][community_type].keys())
            info_pathway_ids_intersection = info_pathway_ids_intersection.intersection(set(community_type_information_pathways[window][community_type].keys()))
        
        for community_type in community_type_information_pathways[window]:
            if not community_type in filtered_community_type_information_pathways:
                filtered_community_type_information_pathways[community_type] = {}
            filtered_community_type_information_pathways[community_type][window] = {tweet_id: community_type_information_pathways[window][community_type][tweet_id] for tweet_id in info_pathway_ids_intersection}

    for community_type in filtered_community_type_information_pathways:
        for window_name in filtered_community_type_information_pathways[community_type]:
            if community_type == "eng":
                # Engagement
                comm_output_dir = "/home/ataylor2/processing_covid_tweets/Thunder/Information_Pathways/top90_Engagement_Comm"
            elif community_type == "int":
                # Interaction
                comm_output_dir = "/home/ataylor2/processing_covid_tweets/Thunder/Information_Pathways/top90_Interaction_Comm"
            elif community_type == "ips":
                # Influence and Passivity
                comm_output_dir = "/home/ataylor2/processing_covid_tweets/Thunder/Information_Pathways/top90_IP_Comm"
            elif community_type == "srd":
                # Semi Random Communities
                comm_output_dir = "/home/ataylor2/processing_covid_tweets/Thunder/Information_Pathways/top90_Semi_Random_Comm"            
            elif community_type == "trd":
                # True Random Communities
                comm_output_dir = "/home/ataylor2/processing_covid_tweets/Thunder/Information_Pathways/top90_True_Random_Comm"

            with open(os.path.join(comm_output_dir, f"{window_name}_instances.pkl"), "wb") as fp:
                dill.dump(filtered_community_type_information_pathways[community_type][window_name], fp)
            
    print("finished")
    print(f"avg ip length: {sum(avg_info_pathway_size)/len(avg_info_pathway_size)}")

if __name__ == "__main__":
    main()