#!/bin/bash

scripts=("ablation_scripts/train_eng_longformer_article.sh"
"ablation_scripts/train_eng_longformer_comm_article.sh"
"ablation_scripts/train_eng_longformer_comm.sh"
"ablation_scripts/train_eng_prim_bert_article.sh"
"ablation_scripts/train_eng_prim_bert_comm_article.sh"
"ablation_scripts/train_eng_prim_bert_comm.sh"
"ablation_scripts/train_eng_prim_clip_article.sh"
"ablation_scripts/train_eng_prim_clip_comm_article.sh"
"ablation_scripts/train_eng_prim_clip_comm.sh"
"ablation_scripts/train_int_longformer_article.sh"
"ablation_scripts/train_int_longformer_comm_article.sh"
"ablation_scripts/train_int_longformer_comm.sh"
"ablation_scripts/train_int_prim_bert_article.sh"
"ablation_scripts/train_int_prim_bert_comm_article.sh"
"ablation_scripts/train_int_prim_bert_comm.sh"
"ablation_scripts/train_int_prim_clip_article.sh"
"ablation_scripts/train_int_prim_clip_comm_article.sh"
"ablation_scripts/train_int_prim_clip_comm.sh"
"ablation_scripts/train_ip_longformer_article.sh"
"ablation_scripts/train_ip_longformer_comm_article.sh"
"ablation_scripts/train_ip_longformer_comm.sh"
"ablation_scripts/train_ip_prim_bert_article.sh"
"ablation_scripts/train_ip_prim_bert_comm_article.sh"
"ablation_scripts/train_ip_prim_bert_comm.sh"
"ablation_scripts/train_ip_prim_clip_article.sh"
"ablation_scripts/train_ip_prim_clip_comm_article.sh"
"ablation_scripts/train_ip_prim_clip_comm.sh"
"ablation_scripts/train_srand_longformer_article.sh"
"ablation_scripts/train_srand_longformer_comm_article.sh"
"ablation_scripts/train_srand_longformer_comm.sh"
"ablation_scripts/train_srand_prim_bert_article.sh"
"ablation_scripts/train_srand_prim_bert_comm_article.sh"
"ablation_scripts/train_srand_prim_bert_comm.sh"
"ablation_scripts/train_srand_prim_clip_article.sh"
"ablation_scripts/train_srand_prim_clip_comm_article.sh"
"ablation_scripts/train_srand_prim_clip_comm.sh")

for script in "${scripts[@]}"; do
    if [[ -x "$script" ]]; then
        nohup ./"$script" > /dev/null 2>&1 &
        echo "Launched $script"
    else
        echo "Error: $script not found or not executable."
    fi
done
