#!/bin/bash

scripts=("train_eng_longformer.sh" "train_eng_prim_bert.sh" "train_eng_prim_clip.sh" 
"train_int_longformer.sh" "train_int_prim_bert.sh" "train_int_prim_clip.sh" 
"train_ip_longformer.sh" "train_ip_prim_bert.sh" "train_ip_prim_clip.sh" 
"train_srand_longformer.sh" "train_srand_prim_bert.sh" "train_srand_prim_clip.sh")

for script in "${scripts[@]}"; do
    if [[ -x "$script" ]]; then
        nohup ./"$script" &
        echo "Launched $script"
    else
        echo "Error: $script not found or not executable."
    fi
done
