import utils
import torch
import os 
from torch_geometric.datasets import Planetoid, Amazon

def compile_separate_embedding_jsons(files_dir, data_window, embedding_dir_path):
    compiled_embeddings = {}
    for embedding_path in data_window:
        embeddings = utils.load_json(os.path.join(files_dir, embedding_path))
        for community in embeddings:
            # print(embeddings[community]["embedding"])
            if not community in compiled_embeddings:
                compiled_embeddings[community] = torch.tensor(embeddings[community]["embedding"])
            else:
                compiled_embeddings[community] = compiled_embeddings[community] + torch.tensor(embeddings[community]["embedding"])
    utils.dump_dill(compiled_embeddings, embedding_dir_path)
     
def filter_tree_files(tree_files):
    data_window_dates_rev = utils.load_data_windows()
    data_window_dates = {vv:k for k,v in data_window_dates_rev.items() for vv in v}
    
    data_windows = {}
    for file in sorted(os.listdir(tree_files)):
        file_date = file.split("_")[0]
        if not data_window_dates[file_date] in data_windows:
            data_windows[data_window_dates[file_date]] = []
        data_windows[data_window_dates[file_date]].append(file) 

    return data_windows
    
def main():
    longformer_embedding_path = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_longformer_embedding"
    primera_bert_embedding_path = "/home/ataylor2/processing_covid_tweets/IP_Analysis/Community_Embeddings/community_primera_bert_embedding"
    embeddings_dir = "/home/ataylor2/processing_covid_tweets/Thunder/compiled_embeddings_TEMP"  

    for files_dir, model_name in [(longformer_embedding_path, "longformer"), (primera_bert_embedding_path, "primera_bert")]:
        # window_name = f'window_{i}'
        window_aggregated_embeds = filter_tree_files(files_dir)
        for window_name in window_aggregated_embeds:
            compile_separate_embedding_jsons(files_dir, window_aggregated_embeds[window_name], os.path.join(embeddings_dir, window_name + "_" + model_name))

def retrieve_tree_files(dir_path):
    files = []
    for filename in os.listdir(dir_path):
        files.append(os.path.join(dir_path, filename))
    return files

def main2():
    article_embeddings_path = "/home/ataylor2/processing_covid_tweets/Thunder/article_embeddings/window_1_article_embeddings.pkl"
    
    article_embeddings = utils.load_dill(article_embeddings_path)
    print(len(article_embeddings))
    max_id = 0
    for article_embedding_id in article_embeddings:
        if article_embedding_id > max_id:
            max_id = article_embedding_id
    
    
    tweet_tree_path = "/home/ataylor2/processing_covid_tweets/info_pathways_full_STRUCTURE"
    data_window_path = "/home/ataylor2/processing_covid_tweets/Thunder/metadata/data_windows_trees.json"
    
    num_sampled_tweets = 8000
    
    tree_files = retrieve_tree_files(tweet_tree_path)
    if not os.path.exists(data_window_path):
        data_windows = utils.filter_tree_files(tree_files)
    else:
        data_windows = utils.load_json(data_window_path)

    pathways = 0
    for i, window in enumerate(data_windows):
       
        if i == 0 or i == 2:
            continue
        for tweet_list_file_path in data_windows[window]:
            if "_reply" in tweet_list_file_path:
                continue
        
            tweet_tree_list = utils.load_dill(tweet_list_file_path)
            pathways += len(tweet_tree_list)
    print(pathways)

def main():
    tweet_tree_list = utils.load_dill("/home/ataylor2/processing_covid_tweets/Thunder/sampled_tweet_trees/window_1_article_mapping.pkl")
    for tweet in tweet_tree_list:
        print(tweet, tweet_tree_list[tweet])
        break

def main3():
    def load_dataset(dataset):
        """
        Load the dataset from PyG.

        :param dataset: name of the dataset. Options: 'Cora', 'CiteSeer', 'PubMed', 'Photo', 'Computers'
        :return: PyG dataset data.
        """
        data_folder = f'data/{dataset}/'
        if dataset in ('Cora', 'CiteSeer', 'PubMed'):
            pyg_dataset = Planetoid(data_folder, dataset)
        elif dataset in ('Photo', 'Computers'):
            pyg_dataset = Amazon(data_folder, dataset)
        else:
            raise NotImplementedError(f'{dataset} not supported. ')
        data = pyg_dataset.data
        return data
    data = load_dataset("Photo")
    print(data)

if __name__ == "__main__":
    main3()