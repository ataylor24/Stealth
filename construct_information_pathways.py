import os, sys, mmap
from tqdm.auto import tqdm
import dill
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import utils

def construct_and_map_information_pathways(tweet_list_file_path, community_assignments_by_comm):
    community_assignments_by_user = utils.reverse_two_layer_dict(community_assignments_by_comm)

    comm_mapping = utils.create_key_mapping(community_assignments_by_comm)
    
    tweet_tree_list = utils.load_dill(tweet_list_file_path)
    information_pathways = {}
    for tweet_tree in tqdm(tweet_tree_list):
        information_pathway= {
            'tweet_ids': {}, 
            'comm_to_comm': set(), 
            'article_to_comm': None, 
            'article_from_news_source': None
        }
         
        for edge, tweet_id in tweet_tree_list[tweet_tree]['tweet_ids'].items():
            try:
                information_pathway['tweet_ids'][(community_assignments_by_user[edge[0]],community_assignments_by_user[edge[1]])] = tweet_id
            except:
                pass
            
        for edge in tweet_tree_list[tweet_tree]['comm_to_comm']:
            try:
                information_pathway['comm_to_comm'].add((community_assignments_by_user[edge[0]],community_assignments_by_user[edge[1]]))
            except:
                pass
            
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
    utils.set_news_urls()
    
    tweet_tree_path = "/home/ataylor2/processing_covid_tweets/Thunder/sampled_tweet_trees"
    
    #eng, int, ips
    community_types = ["eng", "int", "ips"]
    
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
                comm_output_dir = "/home/ataylor2/processing_covid_tweets/Thunder/Information_Pathways/Engagement_Comm"
            elif community_type == "int":
                # Interaction
                comm_output_dir = "/home/ataylor2/processing_covid_tweets/Thunder/Information_Pathways/Interaction_Comm"
            elif community_type == "ips":
                # Influence and Passivity
                comm_output_dir = "/home/ataylor2/processing_covid_tweets/Thunder/Information_Pathways/IP_Comm"
            with open(os.path.join(comm_output_dir, f"{window_name}_instances.pkl"), "wb") as fp:
                dill.dump(filtered_community_type_information_pathways[community_type][window_name], fp)
            
    print("finished")

if __name__ == "__main__":
    main()