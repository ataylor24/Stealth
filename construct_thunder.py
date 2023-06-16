import os, sys, mmap
from tqdm.auto import tqdm
import dill
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import utils
import generate_community_embeddings

pb_device = "cuda:1"
pc_device = "cuda:3"
lf_device = "cuda:6"

def build_community_embeddings_for_window(window_name, window, window_images):
    comm_embeddings_path = "/home/ataylor2/processing_covid_tweets/Thunder/community_embeddings"
   
    aggregated_content_for_communities = {}
    aggregated_ids_for_communities = {}
    for article_data_day_path in window:
        print(f"processing {article_data_day_path}")
        article_data_day = utils.load_json(article_data_day_path)
        for article_data in article_data_day:
            source = article_data['source_name']
            if not source in aggregated_content_for_communities:
                aggregated_content_for_communities[source] = []
            if not source in aggregated_ids_for_communities:
                aggregated_ids_for_communities[source] = []
            
            if 'id' not in article_data:
                continue
            if article_data['id'] in window_images:
                aggregated_ids_for_communities[source].append(window_images[article_data['id']])
            else:
                aggregated_ids_for_communities[source].append(None)
                
            aggregated_content_for_communities[source].append(article_data["title"] + " ".join(article_data["content"]))    
            
    
    # pb_initialization = generate_community_embeddings.initialize_primera_bert()
    pc_initialization = generate_community_embeddings.initialize_primera_clip(pc_device)
    # lf_initialization = generate_community_embeddings.initialize_longformer(lf_device)
    
    embeddings_types = {
        "primera_bert": {},
        "primera_clip": {},
        "longformer": {}
    }
    for community in tqdm(aggregated_content_for_communities):
        if community == "":
            continue
                
        agg_community_content = aggregated_content_for_communities[community]
        agg_community_sources = aggregated_ids_for_communities[community]
        
        unified_content = list(zip(agg_community_content, agg_community_sources))
        random.shuffle(unified_content)
        
        agg_community_content, agg_community_sources = zip(*unified_content)
        
        # pb_output = generate_community_embeddings.get_primera_bert_embedding(*pb_initialization, pb_device, agg_community_content)
        pc_output = generate_community_embeddings.get_primera_clip_embedding(*pc_initialization, pc_device, agg_community_content, agg_community_sources)
        # lf_output = generate_community_embeddings.get_longformer_embedding(*lf_initialization, lf_device, agg_community_content)
        
        # embeddings_types["primera_bert"][community] = pb_output
        embeddings_types["primera_clip"][community] = pc_output
        # embeddings_types["longformer"][community] = lf_output
        
    
    for embeddings_type in embeddings_types:
        utils.dump_dill(embeddings_types[embeddings_type],os.path.join(comm_embeddings_path, window_name + "_" + embeddings_type))
 
def retrieve_article_files(dir_path):
    files = []
    for filename in os.listdir(dir_path):
        if filename != "IMG":
            files.append(os.path.join(dir_path, filename))
    return files

def retrieve_image_files(dir_path):
    files = []
    for subdir in os.listdir(dir_path):
        files.append(os.path.join(dir_path, subdir))
    return files
    
def main():
    utils.set_news_urls()
    
    articles_path = "/home/alvynw/articlesFinal"
    images_path = "/home/alvynw/articlesFinal/IMG"
    data_window_path = "/home/ataylor2/processing_covid_tweets/Thunder/metadata/data_windows.json"
    data_window_images_path = "/home/ataylor2/processing_covid_tweets/Thunder/metadata/data_windows_images.json"

    article_files = retrieve_article_files(articles_path)
    if not os.path.exists(data_window_path):
        data_windows = utils.filter_article_files(article_files)
    else:
        data_windows = utils.load_json(data_window_path)
        
    image_files = retrieve_image_files(images_path)
    if not os.path.exists(data_window_images_path):
        data_windows_images = utils.filter_image_files(image_files)
    else:
        data_windows_images = utils.load_json(data_window_images_path)
    
    for window in data_windows:
        build_community_embeddings_for_window(window, data_windows[window], data_windows_images[window])
        
    print("finished")

if __name__ == "__main__":
    main()