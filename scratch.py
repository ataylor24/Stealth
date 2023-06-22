import utils
import torch
import os 


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

if __name__ == "__main__":
    main()