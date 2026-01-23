import os
import json
import gzip
import pickle
from collections import defaultdict
from tqdm import tqdm
import argparse

from main import allmrec
from pre_train.sasrec.main import sasrec_main
from models.a_llmrec_model import *
import difflib


def load_train_dict_from_txt(train_file_path):
    train_dict = defaultdict(list)
    with open(train_file_path, 'r') as f:
        for line in f:
            user, item = map(int, line.strip().split())
            train_dict[user].append(item)
    return train_dict


def call_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument('--gpu_num', type=int, default=0)
    parser.add_argument("--llm", type=str, default='opt', help='opt, llama')
    parser.add_argument("--recsys", type=str, default='sasrec')
    parser.add_argument("--rec_pre_trained_data", type=str, default='ml-100k')
    parser.add_argument('--batch_size1', default=32, type=int)
    parser.add_argument('--batch_size2', default=2, type=int)
    parser.add_argument('--batch_size_infer', default=2, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument("--stage1_lr", type=float, default=0.0001)
    parser.add_argument("--stage2_lr", type=float, default=0.0001)
    parser.add_argument('--topN', type=int, default=5)                
    args = parser.parse_args()

    print(f"device num : {args.gpu_num}")
    args.device = 'cuda:' + str(args.gpu_num)
    args.rec_pre_trained_data = "ml-100k"
    return args


if __name__ == "__main__":
    data_path   = "A-LLMRec/data/amazon"
    result_path = "A-LLMRec"
    train_txt_path = os.path.join(data_path, 'ml-100k.txt')
    predict_json_path = os.path.join(result_path, "recommend_all_users_filter_bubble_check.json")

    args = call_args()

                    
    train_dict = load_train_dict_from_txt(train_txt_path)

                                                 
                          
    sasrec_main()
    allmrec(args, phase=1)
    allmrec(args, phase=2)

                                      
    args.phase = 3
    model = A_llmrec_model(args).to(args.device)
    model.load_model(args, phase1_epoch=10, phase2_epoch=5)

                     
    with gzip.open(os.path.join(data_path, "ml-100k_text_name_dict.json.gz"), "rb") as f:
        name_dict = pickle.load(f)
    title2id = {v.lower(): k for k, v in name_dict["title"].items()}
    known_titles = list(title2id.keys())

                                    
    all_predict = defaultdict(list)
    for u_id in tqdm(train_dict.keys(), desc="Recommending for all users"):
        user_train = train_dict.get(u_id, [])
        if not user_train:
            continue
        
        existing_ids = set()                    
        tries_per_rec = 5                          
        banned_ids = list(set(user_train))
        for _ in range(args.topN):
                                                
            got_one = False
            for _try in range(tries_per_rec):
                pred_title = allmrec(args, phase=3, user_id=u_id, user_train=user_train, user_gt = banned_ids,model=model)
                if not isinstance(pred_title, str) or not pred_title.strip():
                    continue

                tl = pred_title.strip().lower()

                              
                if tl in title2id:
                    pid = title2id[tl]
                    if (pid not in user_train) and (pid not in existing_ids):
                                                                     
                        all_predict[u_id].append(pid)
                        existing_ids.add(pid)
                        got_one = True
                        break
                    else:
                        continue

                                       
                matches = difflib.get_close_matches(tl, known_titles, n=1, cutoff=0.90)
                if matches:
                    mt = matches[0]
                    pid = title2id[mt]
                    if (pid not in user_train) and (pid not in existing_ids):
                                         
                        canonical = name_dict["title"][pid]
                                                            
                        all_predict[u_id].append(pid)
                        existing_ids.add(pid)
                        got_one = True
                        break
                else:
                    continue

                                              

              
    with open(predict_json_path, "w") as f:
        json.dump({int(k): v for k, v in all_predict.items()}, f, indent=2)

    print(f"✅ Done. Saved recommendations to: {predict_json_path}")
