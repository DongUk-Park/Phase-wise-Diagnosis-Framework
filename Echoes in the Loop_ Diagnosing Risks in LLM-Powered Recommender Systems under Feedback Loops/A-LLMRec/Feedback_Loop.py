#!/usr/bin/env python3
                       

import os
import json
import gzip
import pickle
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from main import allmrec
from pre_train.sasrec.main import sasrec_main
from models.a_llmrec_model import *
import argparse
import difflib
import shutil
            
import os
import pandas as pd

def build_text_files_from_ratings(train_path, label_path, data_dir, fname):
       
    train_raw_txt_path   = os.path.join(data_dir, f"{fname}_train_raw.txt")
    train_txt_path = os.path.join(data_dir, f"{fname}.txt")
    label_txt_path = os.path.join(data_dir, f"{fname}_label.txt")                
                                                                   
    print(f"Processing Train: {train_path} ...")
    df_train = pd.read_csv(train_path, sep="\t", header=None, usecols=[0, 1])
    
                     
    print(f"Processing Label: {label_path} ...")
    df_label = pd.read_csv(label_path, sep="\t", header=None, usecols=[0, 1, 2])                                  
                                  
    df_train.to_csv(train_txt_path, sep=" ", header=False, index=False)
    df_train.to_csv(train_raw_txt_path, sep=" ", header=False, index=False)
    
    df_label.to_csv(label_txt_path, sep=" ", header=False, index=False)

    print(f"✅ Created efficiently:\n  - {train_txt_path}\n  - {label_txt_path}")
    
def get_gt_from_txt(file_path):
       
                            
    df = pd.read_csv(file_path, sep="\t", header=None, names=["user_id", "item_id", "timestamp"])       
                                                   
    ts_num = pd.to_numeric(df["timestamp"], errors="coerce")
    if ts_num.notna().all():
        df["date"] = pd.to_datetime(ts_num.astype("int64"), unit="s").dt.strftime("%Y-%m-%d")
    else:
        df["date"] = df["timestamp"].astype(str)
    
    unique_dates = sorted(df["date"].unique())
    date2idx = {date: idx for idx, date in enumerate(unique_dates)}

    data = defaultdict(list)
                       
    for user_id, group in df.groupby("user_id"):
                                                       
        group = group.sort_values(by="timestamp")
        items = list(zip(group["item_id"], group["date"]))
        for item_id, date in items:
            date_idx = date2idx[date]
            data[user_id].append((item_id, date_idx))
            
    return data, date2idx

def load_train_dict_from_txt(train_file_path):
    train_dict = defaultdict(list)
    with open(train_file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                user = int(parts[0]); item = int(parts[1])
            train_dict[user].append(item)
    return train_dict

def update_train(predict_dict, train_file_path):
    user_history = defaultdict(list)

    if os.path.exists(train_file_path):
        with open(train_file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    user = int(parts[0]); item = int(parts[1])
                    user_history[user].append(item)

    for user, ids in predict_dict.items():
        if isinstance(ids, int):
            ids = [ids]
        for it in ids:
            user_history[int(user)].append(int(it))

    with open(train_file_path, "w") as f:
        for user in sorted(user_history.keys()):
            for item in user_history[user]:
                f.write(f"{user}\t{item}\n")       
                           
def get_data_books(data_root):
       
    BOOKS_DIR         = "data/books"
    BOOKS_META_PATH   = os.path.join(BOOKS_DIR, "item_meta_2017_kcore10_user_item_split_filtered.json")
    BOOKS_TRAIN = os.path.join(BOOKS_DIR, "train.txt")
    BOOKS_LABEL = os.path.join(BOOKS_DIR, "label.txt")
    
                                 
    books_dir = os.path.join(data_root, "books")
    os.makedirs(books_dir, exist_ok=True)
    
                          
    build_text_files_from_ratings(BOOKS_TRAIN, BOOKS_LABEL, books_dir, fname="books")

                 
    train_txt_path = os.path.join(books_dir, "books.txt")
    label_txt_path = os.path.join(books_dir, "books_label.txt")

                                                 
    print(f"Loading train_dict from {train_txt_path} ...")
    train_dict = load_train_dict_from_txt(train_txt_path)
    
    print(f"Loading Ground Truth from {label_txt_path} ...")
    gt, date2idx = get_gt_from_txt(label_txt_path)                         

    return (gt, date2idx), train_dict, "books", train_txt_path

def get_data_ml_1m(data_dir):
       
    ml_1m_dir = os.path.join(data_dir, "ml-1m")
    
    a_llmrec_dir = os.path.join(ml_1m_dir, "A-LLMRec_format")
    os.makedirs(a_llmrec_dir, exist_ok=True)

                 
    train_raw_path = os.path.join(ml_1m_dir, "train.txt")
    label_raw_path = os.path.join(ml_1m_dir, "label.txt")

                                                                                         
    train_txt_path = os.path.join(a_llmrec_dir, "ml-1m.txt")
    label_txt_path = os.path.join(a_llmrec_dir, "ml-1m_label.txt")
    
                             
    
    shutil.copyfile(train_raw_path, train_txt_path)
    shutil.copyfile(label_raw_path, label_txt_path)

                                                 
    print(f"Loading train_dict from {train_txt_path} ...")
    train_dict = load_train_dict_from_txt(train_txt_path)
    
    print(f"Loading Ground Truth from {label_txt_path} ...")
    gt, date2idx = get_gt_from_txt(label_txt_path)                         

    return (gt, date2idx), train_dict, "ml-1m", train_txt_path

                               
def call_args():
    parser = argparse.ArgumentParser()

         
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument('--gpu_num', type=int, default=0)

                   
    parser.add_argument("--llm", type=str, default='opt', help='opt, llama')
    parser.add_argument("--recsys", type=str, default='sasrec')

                     
    parser.add_argument("--rec_pre_trained_data", type=str, default='ml-1m', choices=['ml-1m','books'],
                        help='ml-1m | books')

    parser.add_argument("--item_num", type=int, default=3693, help="books:25882, ml-1m:3693")

    parser.add_argument("--train_txt_path", type=str, help='ml-1m | books')
                         
    parser.add_argument('--phase', type=int, default=0)

                             
    parser.add_argument('--batch_size1', default=32, type=int)
    parser.add_argument('--batch_size2', default=2, type=int)
    parser.add_argument('--batch_size_infer', default=2, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument("--stage1_lr", type=float, default=0.0001)
    parser.add_argument("--stage2_lr", type=float, default=0.0001)
    
    print("ENV CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    args = parser.parse_args()
    print(f"device num : {args.gpu_num}")
    args.device = 'cuda:' + str(args.gpu_num)
    return args
                           
if __name__ == "__main__":
                      
    args = call_args()
    dataset = args.rec_pre_trained_data                      
    
    DATA_ROOT   = "data/"
    RESULT_ROOT = "data/"+dataset+"/A-LLMRec_format/A-LLMRec_results"

                       
    os.makedirs(RESULT_ROOT, exist_ok=True)
    open(os.path.join(RESULT_ROOT, "candidate_debug.jsonl"), "w").close()
         
    cases = [2]
    parts = [1]
                         
    if dataset == "ml-1m":
        gt, _, fname, train_txt_path = get_data_ml_1m(DATA_ROOT)
    elif dataset == "books":
        gt, _, fname, train_txt_path = get_data_books(DATA_ROOT)

    ground_truth, date2idx = gt
    data_dir = os.path.join(DATA_ROOT, fname)
    os.makedirs(data_dir, exist_ok=True)

    args.train_txt_path = train_txt_path
    predict_json_path_tmpl = os.path.join(RESULT_ROOT, "predict_label_{tag}.json")
    
             
    all_timestamps = sorted(set(ts for interactions in ground_truth.values() for _, ts in interactions))

    for ca in cases:
        for part in parts:
            tag = f"part{part}"
            print(f"{tag} start, do update_train")
            predict_json_path = predict_json_path_tmpl.format(tag=tag)

            part_size = max(1, len(all_timestamps) // part)
            all_parts_predict_label = defaultdict(list)

            for t in range(part):
                print(f"\n▶ [{dataset}] Part {t+1}/{part} 시작")
                predict_label = defaultdict(list)
                missing_titles_by_user = defaultdict(list)
                duplicates_by_user = defaultdict(list)

                             
                train_dict = load_train_dict_from_txt(args.train_txt_path)

                                     
                exp_tag = f"{fname}_case{ca}_parts{part}"
                os.environ["SASREC_PART_ID"] = str(t)
                os.environ["SASREC_EMB_DIR"] = f"data/{dataset}/A-LLMRec_format/u_i_embedding_in_sasrec/{exp_tag}"

                            
                if t >= 0:
                    sasrec_main(dataset=dataset, path=args.train_txt_path, item_num = args.item_num)
                    allmrec(args, phase=1)  
                    allmrec(args, phase=2)

                           
                start_idx = t * part_size
                end_idx   = (t + 1) * part_size if t < part - 1 else len(all_timestamps)
                time_range = set(all_timestamps[start_idx:end_idx])
                
                                                                     
                common_users = set(train_dict.keys()) & set(ground_truth.keys())

                                                                     
                active_users = [
                    u for u in common_users
                    if any(ts in time_range for _, ts in ground_truth[u])
                ]

                                                      
                user_interaction_count = {
                    u: sum(1 for _, ts in ground_truth[u] if ts in time_range)
                    for u in active_users
                }

                            
                args.phase = 3
                model = A_llmrec_model(args).to(args.device)
                model.load_model(args, phase1_epoch=10, phase2_epoch=5)

                        
                for u_id in tqdm(user_interaction_count.keys(), desc=f"[{dataset}] Generating predictions"):
                    user_train = train_dict.get(u_id, [])
                    user_train_set = set(user_train)

                    interactions_in_part = [(itm, ts) for (itm, ts) in ground_truth[u_id] if ts in time_range]
                    interactions_in_part.sort(key=lambda x: x[1])

                    if len(user_train) == 0 or len(interactions_in_part) == 0:
                        continue

                    need = min(user_interaction_count[u_id], len(interactions_in_part))
                    if need == 0:
                        continue

                    existing_ids = set()

                    for k in range(need):
                        pos_id = int(interactions_in_part[k][0])
                        banned_ids = list(user_train_set)

                        predicted_title, candidates_include_target_ids, candidates_include_target_titles = allmrec(
                            args,
                            pos_item_id=pos_id,
                            phase=3,
                            user_id=u_id,
                            user_train=user_train,
                            user_gt=banned_ids,
                            model=model
                        )

                        if not isinstance(predicted_title, str) or not predicted_title.strip():
                            missing_titles_by_user[u_id].append(f"<INVALID:{repr(predicted_title)}>")
                            continue

                        title_lower = predicted_title.strip().lower()
                        candidates_include_target_titles = [s.strip('"').lower() for s in candidates_include_target_titles]

                                 
                        if title_lower in candidates_include_target_titles:
                                                                              
                            predicted_id = int(candidates_include_target_ids[candidates_include_target_titles.index(title_lower)])                           
                            if (predicted_id not in user_train_set) and (predicted_id not in existing_ids):
                                predict_label[u_id].append(predicted_id)
                                existing_ids.add(predicted_id)
                            else:
                                duplicates_by_user[u_id].append(predicted_id)
                            continue

                        matches = difflib.get_close_matches(title_lower, candidates_include_target_titles, n=1, cutoff=0.85)
                        if matches:
                            full_title = matches[0]
                            predicted_id = int(candidates_include_target_ids[candidates_include_target_titles.index(full_title)])                           
                            if (predicted_id not in user_train_set) and (predicted_id not in existing_ids):
                                predict_label[u_id].append(predicted_id)
                                existing_ids.add(predicted_id)
                            else:
                                duplicates_by_user[u_id].append(predicted_id)
                        else:
                            missing_titles_by_user[u_id].append(predicted_title.strip())

                           
                for u_id, ids in predict_label.items():
                    all_parts_predict_label[u_id].extend(ids)

                       
                missing_out_path = os.path.join(RESULT_ROOT, f"missing_titles_{fname}_case{ca}_part{part}_step{t+1}.json")
                with open(missing_out_path, "w") as f:
                    json.dump({int(k): v for k, v in missing_titles_by_user.items()}, f, indent=2)
                print(f"Missing titles saved: {missing_out_path}")

                duplicates_out_path = os.path.join(RESULT_ROOT, f"skipped_duplicates_{fname}_case{ca}_part{part}_step{t+1}.json")
                with open(duplicates_out_path, "w") as f:
                    json.dump({int(k): v for k, v in duplicates_by_user.items()}, f, indent=2)
                print(f"Duplicates skipped saved: {duplicates_out_path}")

                            
                update_train(predict_label, args.train_txt_path)
                print(f"[{dataset}] Part {t+1}/{part} 완료")

                temp_predict_path = os.path.join(
                    RESULT_ROOT,
                    f"predict_label_part{part}_step{t+1}.json"
                )
                with open(temp_predict_path, "w") as f:
                    json.dump({int(k): v for k, v in predict_label.items()}, f, indent=2)

                   
            with open(predict_json_path, "w") as f:
                json.dump(all_parts_predict_label, f, indent=2)
            print(f"✅ Saved predictions: {predict_json_path}")
