import json
import os
import dill

def create_key_mapping(original_dict):
    dict_mapping = {}
    for i, key in enumerate(original_dict.keys()):
        dict_mapping[key] = i
    return dict_mapping

def reverse_one_layer_dict(original_dict):
    return {v: k for k,v in original_dict.items()}

def reverse_two_layer_dict(original_dict):
    return {kk: k for k,v in original_dict.items() for kk, vv in v.items()}

def dump_dill(dump_obj, filepath):
    dill.dump(dump_obj, open(filepath, 'wb'))

def load_dill(data_path):
    return dill.load(open(data_path, 'rb'))

def retrieve_url_source(url):
    url_prefix = url[:8]
    url_suffix = url[8:].split('/')[0]
    return url_prefix + url_suffix

def retrieve_news_org(url):
    if not isinstance(url, str): 
        if url[0]:
            src_url = retrieve_url_source(url[0])
            news_orgs = news_prefix_urls[src_url]
    else:
        src_url = retrieve_url_source(url)
        news_orgs = news_prefix_urls[src_url]

    return news_orgs

def url_check(source):
    if "http" in source:
        return retrieve_news_org(source)
    else:
        return source

def load_json(filepath):
    return json.load(open(filepath, 'r'))

def filter_tree_files(tree_files):
    data_window_dates_rev = load_data_windows()
    data_window_dates = {vv:k for k,v in data_window_dates_rev.items() for vv in v}
    
    data_windows = {}
    for file in sorted(tree_files):
        file_date = file.split("_")[-1].split(".")[0]
        if not data_window_dates[file_date] in data_windows:
            data_windows[data_window_dates[file_date]] = []
        data_windows[data_window_dates[file_date]].append(file) 

    json.dump(data_windows, open("/home/ataylor2/processing_covid_tweets/Thunder/metadata/data_windows_trees.json", "w"))

def filter_image_files(image_files):
    data_window_dates_rev = load_data_windows()
    data_window_dates = {vv:k for k,v in data_window_dates_rev.items() for vv in v}
    
    data_windows = {}
    for day_path in sorted(image_files):
        file_date = day_path.split("_")[-1]
        
        if not data_window_dates[file_date] in data_windows:
            data_windows[data_window_dates[file_date]] = {}
        
        for article_dir in os.listdir(day_path): 
            data_windows[data_window_dates[file_date]][article_dir] = os.path.join(day_path, article_dir)
            
            
    json.dump(data_windows, open("/home/ataylor2/processing_covid_tweets/Thunder/metadata/data_windows_images.json", "w"))

def filter_article_files(article_files):
    data_window_dates_rev = load_data_windows()
    data_window_dates = {vv:k for k,v in data_window_dates_rev.items() for vv in v}
    
    data_windows = {}
    for file in sorted(article_files):
        file_date = file.split("_")[-1].split(".")[0]
        if not data_window_dates[file_date] in data_windows:
            data_windows[data_window_dates[file_date]] = []
        data_windows[data_window_dates[file_date]].append(file) 
    
    json.dump(data_windows, open("/home/ataylor2/processing_covid_tweets/Thunder/metadata/data_windows.json", "w"))
        
def load_data_windows():
    return load_json("/home/ataylor2/processing_covid_tweets/Thunder/metadata/data_windows_dates.json")

def set_news_urls():
    news_urls_dict = "/home/ataylor2/processing_covid_tweets/IP_Analysis/IP_analysis_news_url_mapping.json"
    
    with open(news_urls_dict, "r") as f:
        news_prefix_urls_rev = json.loads(f.read())
    # print(news_prefix_urls_rev)
    global source_names
    source_names = news_prefix_urls_rev
    global news_prefix_urls
    news_prefix_urls = {v: k for k, v in news_prefix_urls_rev.items()}
