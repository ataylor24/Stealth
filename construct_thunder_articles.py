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


clip_device = "cuda:3"
lf_device = "cuda:5"

def build_article_embeddings_for_window(window_name, window, window_images, window_article_mapping, model_name):
    
    article_embeddings = {}
    # pc_initialization = generate_community_embeddings.initialize_primera_clip(pc_device)
    if model_name == "longformer":
        lf_initialization = generate_community_embeddings.initialize_longformer(lf_device)
    elif model_name == "clip":
        clip_initialization = generate_community_embeddings.initialize_clip(clip_device)
    
    for article_data_day_path in window:
        print(f"processing {article_data_day_path}")
        article_data_day = utils.load_json(article_data_day_path)
        for article_data in tqdm(article_data_day):
            if article_data["source_url"] in window_article_mapping:
                if model_name == "longformer":
                    article_embeddings[window_article_mapping[article_data["source_url"]]] = \
                        generate_community_embeddings.get_longformer_embedding(*lf_initialization, lf_device, article_data['content'])
                elif model_name == "clip":
                    if 'id' in article_data and article_data['id'] in window_images:
                        image_dir = window_images[article_data['id']]
                    else:
                        image_dir = None
                    try:
                        article_embeddings[window_article_mapping[article_data["source_url"]]] = \
                            generate_community_embeddings.get_clip_embedding(*clip_initialization, clip_device, article_data['content'], image_dir)
                    except RuntimeError:
                        pass
                
    return article_embeddings

def retrieve_article_mappings(dir_path):
    files = {}
    for file_ in os.listdir(dir_path):
        if "article_mapping" in file_:
            files[file_[:8]] = utils.load_dill(os.path.join(dir_path, file_))
    return files

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
    
    model_name = "clip"
    
    if model_name == "longformer":
        article_embeddings_output_path = "/home/ataylor2/processing_covid_tweets/Thunder/article_embeddings"
    elif model_name == "clip":
        article_embeddings_output_path = "/home/ataylor2/processing_covid_tweets/Thunder/article_image_embeddings"

    articles_path = "/home/alvynw/articlesFinal"
    images_path = "/home/alvynw/articlesFinal/IMG"
    data_window_path = "/home/ataylor2/processing_covid_tweets/Thunder/metadata/data_windows.json"
    data_window_images_path = "/home/ataylor2/processing_covid_tweets/Thunder/metadata/data_windows_images.json"

    sampled_articles_path = "/home/ataylor2/processing_covid_tweets/Thunder/sampled_tweet_trees"
    article_mappings = retrieve_article_mappings(sampled_articles_path)
    
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
    
    for i, window_name in enumerate(data_windows):
        if i == 0:
            continue
        window_articles_embedding = build_article_embeddings_for_window(window_name, data_windows[window_name], data_windows_images[window_name], utils.reverse_one_layer_dict(article_mappings[window_name]), model_name)
        utils.dump_dill(window_articles_embedding, os.path.join(article_embeddings_output_path, f"{window_name}_article_embeddings.pkl"))
        
    
    print("finished")

if __name__ == "__main__":
    main()