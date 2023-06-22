from collections import defaultdict
from transformers import AutoTokenizer, LEDForConditionalGeneration, LongformerConfig, \
    LongformerTokenizerFast, LongformerModel
import random
from transformers import BertTokenizer, BertModel
import torch
import os
import json
import yaml
import clip
import argparse
from PIL import Image
import logging

logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
# device = "cuda:0"#torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

def initialize_primera_bert():
    # Load Primera ModelNumber of documents
    TOKENIZER = AutoTokenizer.from_pretrained('allenai/PRIMERA')
    MODEL = LEDForConditionalGeneration.from_pretrained('allenai/PRIMERA')

    DOCSEP_TOKEN_ID = TOKENIZER.convert_tokens_to_ids("<doc-sep>")

    # Load pre-trained model tokenizer (vocabulary)
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Load pre-trained model (weights)
    bert_model = BertModel.from_pretrained('bert-base-uncased',
                                           output_hidden_states=True,  # Whether the model returns all hidden-states.
                                           )
    return TOKENIZER, MODEL, DOCSEP_TOKEN_ID, bert_tokenizer, bert_model

def initialize_primera_clip(device):
    print("Load Primera Model")
    TOKENIZER = AutoTokenizer.from_pretrained('allenai/PRIMERA')
    MODEL = LEDForConditionalGeneration.from_pretrained('allenai/PRIMERA')

    DOCSEP_TOKEN_ID = TOKENIZER.convert_tokens_to_ids("<doc-sep>")

    print("Load CLIP Model")
    # Load CLIP Model
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    return TOKENIZER, MODEL, DOCSEP_TOKEN_ID, clip_model, preprocess

def initialize_clip(device):
    print("Load CLIP Model")
    # Load CLIP Model
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    
    return clip_model, preprocess

def initialize_longformer(device):
    tokenizer = LongformerTokenizerFast.from_pretrained("allenai/longformer-base-4096")
    model = LongformerModel.from_pretrained("allenai/longformer-base-4096").to(device)
    return tokenizer, model

# Create directory if not exists
def create_directory(file_dir):
    # Check whether the specified path exists or not
    isExist = os.path.exists(file_dir)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(file_dir)
        print("The new directory" + file_dir + " is created!")


# Function to load yaml configuration file
def load_config(config_name):
    # folder to load config file
    config_path = "./config/"
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    return config


# preprocess the raw files into a dict with key is the community name and value is the collections of news
def get_community_content_dict(file):
    community_content_of_the_day = defaultdict(list)

    print("processing file:", file)
    articles = json.load(open(file, "rb"))
    print("number of articles:", len(articles))

    for article_dict in articles:
        comm = article_dict['source_name']
        community_content_of_the_day[comm].append(article_dict['content'])

    return community_content_of_the_day

def preprocess_longformer_document(all_docs):
    return " ".join(all_docs)

def preprocess_clip_document(all_docs):
    return " ".join(all_docs)

# tokenize the community content
def process_primera_document(tokenizer, all_docs):
    DOCSEP_TOKEN_ID = tokenizer.convert_tokens_to_ids("<doc-sep>")
    for i, doc in enumerate(all_docs):
        doc = " ".join(doc)
        all_docs[i] = doc

    input_ids = []
    # doc_max_size = 40
    
    block_size = max(4096 // len(all_docs), 100)
    for doc in all_docs:
        input_ids.extend(
            tokenizer.encode(
                doc,
                truncation=True,
                max_length=block_size,
            )[1:-1]
        )
        input_ids.append(DOCSEP_TOKEN_ID)
    input_ids = input_ids[:4094]
    input_ids = (
            [tokenizer.bos_token_id]
            + input_ids
            + [tokenizer.eos_token_id]
    )
    return torch.tensor(input_ids).unsqueeze(0)


def get_bert_embedding(bert_tokenizer, bert_model, text):
    results = bert_tokenizer(text, max_length=512, truncation=True, add_special_tokens=True)
    input_ids = results.input_ids
    attn_mask = results.attention_mask

    input_ids = torch.tensor(input_ids)
    attn_mask = torch.tensor(attn_mask)

    input_ids = input_ids.unsqueeze(0)
    attn_mask = attn_mask.unsqueeze(0)

    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attn_mask)
        hidden_states = outputs[2]
        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)

    return sentence_embedding


def get_primera_bert_embedding(TOKENIZER, MODEL, DOCSEP_TOKEN_ID, bert_tokenizer, bert_model, device, content):
    
    # MODEL.gradient_checkpointing_enable()
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    bert_model.eval()

    cnt_ids = process_primera_document(TOKENIZER, content)
    attention_mask = torch.ones(cnt_ids.shape, dtype=torch.long)  # no MaskedLM
    global_attention_mask = torch.zeros_like(cnt_ids).to(cnt_ids.device)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1
    global_attention_mask[cnt_ids == DOCSEP_TOKEN_ID] = 1
    generated_ids = MODEL.generate(input_ids=cnt_ids, attention_mask=attention_mask,
                                   global_attention_mask=global_attention_mask, max_length=512)
    generated_str = TOKENIZER.batch_decode(
        generated_ids.tolist(), skip_special_tokens=True
    )[0]
    text_embedding = get_bert_embedding(bert_tokenizer, bert_model, generated_str)
    output = {'summary_text': generated_str, 'embedding': text_embedding.tolist()}
    return output


def get_primera_clip_embedding(TOKENIZER, MODEL, DOCSEP_TOKEN_ID, clip_model, preprocess, device, content, image_dirs):
    cnt_ids = process_primera_document(TOKENIZER, list(content))
    attention_mask = torch.ones(cnt_ids.shape, dtype=torch.long)  # no MaskedLM
    global_attention_mask = torch.zeros_like(cnt_ids).to(cnt_ids.device)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1
    global_attention_mask[cnt_ids == DOCSEP_TOKEN_ID] = 1
    generated_ids = MODEL.generate(input_ids=cnt_ids, attention_mask=attention_mask,
                                   global_attention_mask=global_attention_mask, max_length=60)
    generated_str = TOKENIZER.batch_decode(
        generated_ids.tolist(), skip_special_tokens=True
    )[0]
    text = clip.tokenize([generated_str]).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text)

        image_embedding = None
        if 4096 // len(content) > 100:
            #less than 40 docs
            for image_dir in image_dirs:
                if image_dir == None:
                    continue
                for image_name in os.listdir(image_dir):
                    if not 'jpeg' in image_name:
                        continue
                    try:
                        image = preprocess(Image.open(os.path.join(image_dir, image_name))).unsqueeze(0).to(device)
                        image_features = clip_model.encode_image(image)
                    except OSError:
                        print(f"Error loading image: {image_path}")
                        continue
                    if image_embedding == None:
                        image_embedding = image_features
                    else:
                        image_embedding = image_embedding + image_features
        else:
            #more than 40 docs
            for i, image_dir in enumerate(image_dirs):
                if image_dir == None:
                    continue
                if i > 40:
                    break
                for image_name in os.listdir(image_dir):
                    if not 'jpeg' in image_name:
                        continue
                    try:
                        image = preprocess(Image.open(os.path.join(image_dir, image_name))).unsqueeze(0).to(device)
                        image_features = clip_model.encode_image(image)
                    except OSError:
                        print(f"Error loading image: {os.path.join(image_dir, image_name)}")
                        continue
                    if image_embedding == None:
                        image_embedding = image_features
                    else:
                        image_embedding = image_embedding + image_features
    
    clip_embedding = text_embedding + image_embedding if image_embedding != None else torch.zeros(text_embedding.size())
    
    output = {'summary_text': generated_str, 'embedding': clip_embedding.squeeze(0).tolist()}
    return output

def get_clip_embedding(clip_model, preprocess, device, content, image_dir):
    content = preprocess_clip_document(content)
    
    text = clip.tokenize([content[:60]]).to(device)
  
        
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text)

        image_embedding = None
        if image_dir != None:
            for image_name in os.listdir(image_dir):
                if not 'jpeg' in image_name:
                    continue
                image = preprocess(Image.open(os.path.join(image_dir, image_name))).unsqueeze(0).to(device)
                image_features = clip_model.encode_image(image)
                if image_embedding == None:
                    image_embedding = image_features
                else:
                    image_embedding = image_embedding + image_features
        
        clip_embedding = text_embedding + image_embedding if image_embedding != None else torch.zeros(text_embedding.size())
        
    return clip_embedding.squeeze(0).tolist()

def get_longformer_embedding(tokenizer, model, device, content):
    
    content = preprocess_longformer_document(content)
    cnt_ids = torch.tensor(tokenizer.encode(content, max_length=4096, truncation=True)).unsqueeze(0).to(device)
    attention_mask = torch.ones(cnt_ids.shape, dtype=torch.long, device=device)  # no MaskedLM
    global_attention_mask = torch.zeros(cnt_ids.shape, dtype=torch.long,
                                        device=device)  # initialize to global attention to be deactivated for all tokens
    global_attention_mask[:, 0] = 1  # set global attention based on the task.
    with torch.no_grad():
        outputs = model(cnt_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
    generated_embedding = outputs.pooler_output.squeeze(0).cpu()
    return generated_embedding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model")
    parser.add_argument("-text")

    args = parser.parse_args()
    model = args.model

    # config = load_config("my_config.yaml")
    #
    # file_dir = config['file_dir']
    # files_to_process = sorted([os.path.join(file_dir, filename) for filename in os.listdir(file_dir) if
    #                            ".json" in filename])
    model = args.model
    text = args.text
    print(text)

    if model == "Longformer":
        print('Generating embedding using Longformer')
        output = get_longformer_embedding(text)
    elif model == "Primera_Bert":
        print('Generating embedding using Primera and Bert')
        output = get_primera_bert_embedding(text)
    elif model == "Primera_CLIP":
        print('Generating embedding using Primera and CLIP')
        output = get_primera_clip_embedding(text)

    print(output)


if __name__ == "__main__":
    main()