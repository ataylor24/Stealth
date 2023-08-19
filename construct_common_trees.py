import os, sys, mmap
from tqdm.auto import tqdm
import dill
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import utils

def sample_and_construct_tweets(window_name, window, output_dir, num_sampled_tweets, sampling_method):

    user_user_edge_instances = {}
    article_user_edge_instances = {}
    article_newssource_edge_instances = {}
    tweet_ids_per_instance = {}
    information_source_embeddings = []
    
    article_mapping = {}
    article_index = 0
    
    
    
    for tweet_list_file_path in window:
        if "_reply" in tweet_list_file_path:
            continue
        
        tweet_tree_list = utils.load_dill(tweet_list_file_path)
        
        for tweet_id in tqdm(tweet_tree_list.keys()):
            tweet_tree = set()
            tweet_ids = {}
            inst_written_by = None
            inst_mentioned_by = None
            
            skip_tweet = False
            for tweet in tweet_tree_list[tweet_id]:
                if tweet[-1] == 'information_tweet':
                    inst_written_by = [article_index, utils.url_check(tweet[0])]
                    
                    inst_mentioned_by = [utils.url_check(tweet[0]), tweet[1]]
                    
                    article_mapping[article_index] = tweet[0]
                    article_index += 1
                    continue

                user_user_edge = (tweet[0], tweet[1])
                
                
                if user_user_edge not in tweet_ids:
                    tweet_ids[user_user_edge] = set()
                tweet_ids[user_user_edge].add(tweet[2])
                
                tweet_tree.add(user_user_edge)
          
            if skip_tweet:
                continue 
            user_user_edge_instances[tweet_id] = list(tweet_tree)
            article_user_edge_instances[tweet_id] = inst_mentioned_by
            article_newssource_edge_instances[tweet_id] = inst_written_by
            tweet_ids_per_instance[tweet_id] = tweet_ids
    
    if sampling_method == "full":
        randomly_sampled_tweets = user_user_edge_instances.keys() #random.sample(user_user_edge_instances.keys(), num_sampled_tweets)
    elif sampling_method == "sampled":
        randomly_sampled_tweets = random.sample(user_user_edge_instances.keys(), num_sampled_tweets)

    # reply handling
    reply_mapping = {}
    for tweet_id in randomly_sampled_tweets:
        for branch_tweet_id in tweet_ids_per_instance[tweet_id]:
            reply_mapping[branch_tweet_id] = tweet_id
    
    # user_user_edge_instances[tweet_id] = list(tweet_tree)
    # article_user_edge_instances[tweet_id] = inst_mentioned_by
    # article_newssource_edge_instances[tweet_id] = inst_written_by
    # tweet_ids_per_instance[tweet_id] = tweet_ids
    
    for tweet_list_file_path in window:
        if "_nonreply" in tweet_list_file_path:
            continue
        tweet_tree_list = utils.load_dill(tweet_list_file_path)
        
        skip_tweet = False
        for tweet_tree_partial_tuple in tweet_tree_list:
            tweet_tree_partial, tweet_tree_ids = tweet_tree_partial_tuple
            #[('jbreedsker', 'TrewToFoam', '1263627495238180864', 'reply_tweet')]
            user_reply_edge = (tweet_tree_partial[0][0], tweet_tree_partial[0][1])
            
            if tweet_tree_ids[0] in reply_mapping:
                root_tweet_id = reply_mapping[tweet_tree_ids[0]]
                reply_mapping[tweet_tree_ids[1]] = root_tweet_id
                
                if user_user_edge not in tweet_ids:
                    tweet_ids_per_instance[root_tweet_id][user_reply_edge] = set()
                tweet_ids_per_instance[root_tweet_id][user_reply_edge].add(tweet_tree_partial[0][2])
                user_user_edge_instances[root_tweet_id].append(user_reply_edge)

    enumerated_g_instances = {}
    for i,tweet_instance in enumerate(randomly_sampled_tweets):
        
        enumerated_g_instances[i] = {
            "tweet_ids": tweet_ids_per_instance[tweet_instance],
            "comm_to_comm": user_user_edge_instances[tweet_instance],
            "article_to_comm": article_user_edge_instances[tweet_instance],
            "article_from_news_source": article_newssource_edge_instances[tweet_instance]
        }

    file_name = tweet_list_file_path.split("_")[-1][:-4]
    
    with open(os.path.join(output_dir, f"{window_name}_instances.pkl"), "wb") as fp:
        dill.dump(enumerated_g_instances, fp)
    with open(os.path.join(output_dir, f"{window_name}_article_mapping.pkl"), "wb") as fp:
        dill.dump(article_mapping, fp)
    
            
def retrieve_tree_files(dir_path):
    files = []
    for filename in os.listdir(dir_path):
        files.append(os.path.join(dir_path, filename))
    return files

    
def main():
    utils.set_news_urls()
    
    tweet_tree_path = "/home/ataylor2/processing_covid_tweets/info_pathways_full_STRUCTURE"
    data_window_path = "/home/ataylor2/processing_covid_tweets/Thunder/metadata/data_windows_trees.json"
    
    sampling_method = "sampled"
    tweet_tree_output_path = f"/home/ataylor2/processing_covid_tweets/Thunder/{sampling_method}_tweet_trees"
    num_sampled_tweets = 8000
    
    tree_files = retrieve_tree_files(tweet_tree_path)
    if not os.path.exists(data_window_path):
        data_windows = utils.filter_tree_files(tree_files)
    else:
        data_windows = utils.load_json(data_window_path)
    
    for i, window_name in enumerate(data_windows):
        if i == 0:
            continue
        sample_and_construct_tweets(window_name, data_windows[window_name], tweet_tree_output_path, num_sampled_tweets, sampling_method)
        
    print("finished")

if __name__ == "__main__":
    main()