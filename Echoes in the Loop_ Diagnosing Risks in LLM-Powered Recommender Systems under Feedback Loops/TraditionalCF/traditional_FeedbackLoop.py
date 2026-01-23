#!/usr/bin/env python3
                       

   

from __future__ import annotations
import os, sys, json, random, argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import torch

                              
DATA_ROOT = "data/ml-1m"
                         
TRAIN_RAW_FILE = "train.txt"                                                           
LABEL_RAW_FILE = "label.txt"                                                           

TRAIN_REL  = "traditionalCF/train.json"                                               
LABEL_REL  = "traditionalCF/label.json"                                               
PRED_DIR   = "traditionalCF"                                     
RESULTS_DIR = os.path.join(DATA_ROOT, PRED_DIR)

                                           
NUM_ITEMS_TOTAL = 0 
NUM_USERS_TOTAL = 0
PARTS = 5

                                   
sys.path.append("TraditionalCF")
from LightGCN import train_lightgcn_softmax

                    
       
                    
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _ensure_int_dict(d: Dict) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for k, v in d.items():
        try: ki = int(k)
        except: continue
        if isinstance(v, list):
            out[ki] = [int(x) for x in v]
        else:
            try: out[ki] = [int(v)]
            except: out[ki] = []
    return out

def _save_json(path: str, d: Dict[int, List[int]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in d.items()}, f)

def _load_json(path: str) -> Dict[int, List[int]]:
    with open(path, "r") as f:
        raw = json.load(f)
    return _ensure_int_dict(raw)

                    
              
                    
def get_gt(df: pd.DataFrame):
                                                                       
    df = df.copy()
    if df["timestamp"].dtype != object:
        df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime('%Y-%m-%d')
    else:
        df["date"] = df["timestamp"]                  
    unique_dates = sorted(df["date"].unique())
    date2idx = {date: idx for idx, date in enumerate(unique_dates)}

    data = {}
    for user_id, group in df.groupby("user_id"):
        pairs = []
        for _, row in group.iterrows():
            item_id = int(row["item_id"])
            date_idx = date2idx[row["date"]]
            pairs.append((item_id, date_idx))
        data[int(user_id)] = pairs
    return data, date2idx

def get_initial_data_from_files():
       
    train_raw_path = os.path.join(DATA_ROOT, TRAIN_RAW_FILE)
    label_raw_path = os.path.join(DATA_ROOT, LABEL_RAW_FILE)
    
    if not os.path.exists(train_raw_path) or not os.path.exists(label_raw_path):
        raise FileNotFoundError(f"Check files: {train_raw_path}, {label_raw_path}")

                                                         
    print(f"Loading raw files from {DATA_ROOT}...")
    train_df = pd.read_csv(train_raw_path, sep=r"\s+", names=["user_id", "item_id", "timestamp"])
    label_df = pd.read_csv(label_raw_path, sep=r"\s+", names=["user_id", "item_id", "timestamp"])

                                    
    global NUM_USERS_TOTAL, NUM_ITEMS_TOTAL
    max_u = max(train_df["user_id"].max(), label_df["user_id"].max())
    max_i = max(train_df["item_id"].max(), label_df["item_id"].max())
    NUM_USERS_TOTAL = int(max_u) + 1
    NUM_ITEMS_TOTAL = int(max_i) + 1
    print(f"Detected Stats: Users={NUM_USERS_TOTAL}, Items={NUM_ITEMS_TOTAL}")

                
    train_json_path = os.path.join(DATA_ROOT, TRAIN_REL)
    label_json_path = os.path.join(DATA_ROOT, LABEL_REL)

                   
    train_dict: Dict[int, List[int]] = {int(uid): [int(x) for x in g["item_id"].tolist()]
                                        for uid, g in train_df.groupby("user_id")}
    _save_json(train_json_path, train_dict)

                   
    label_dict: Dict[int, List[int]] = {int(uid): [int(x) for x in g["item_id"].tolist()]
                                        for uid, g in label_df.groupby("user_id")}
    _save_json(label_json_path, label_dict)

    print(f"✅ Saved train.json({len(train_df)}) & label.json({len(label_df)}) at {os.path.dirname(train_json_path)}")

                                 
    test_data_timestamp, date2idx = get_gt(label_df)
    return test_data_timestamp, date2idx

                    
                             
                    
@torch.no_grad()
def recommend_and_update_by_datewindow(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    train_dict: Dict[int, List[int]],
    gt_by_dateidx: Dict[int, List[Tuple[int, int]]],
    date_idx_range: Set[int],
) -> Dict[int, List[int]]:
    
    device = user_emb.device
    pred_dict: Dict[int, List[int]] = {}
    seen_cache: Dict[int, set] = {u: set(items) for u, items in train_dict.items()}

                             
    active_users = [u for u, pairs in gt_by_dateidx.items()
                    if any(di in date_idx_range for _, di in pairs)]

    for u in active_users:
                                  
        Kmax_u = sum(1 for (_it, di) in gt_by_dateidx.get(u, []) if di in date_idx_range)
        if Kmax_u <= 0:
            pred_dict[u] = []
            continue

                  
        scores = (user_emb[u:u+1] @ item_emb.T).squeeze(0)                     
        
                      
        seen = train_dict.get(u, [])
        if seen:
            seen_idx = torch.tensor(seen, dtype=torch.long, device=device)
            scores.index_fill_(0, seen_idx, -1e9)

        num_unseen = int((scores > -1e9/2).sum().item())
        k = max(0, min(Kmax_u, num_unseen))
        if k == 0:
            pred_dict[u] = []
            continue

        topk_idx = torch.topk(scores, k=k).indices.tolist()

                               
        seen_set = seen_cache.get(u, set())
        new_items = [int(x) for x in topk_idx if x not in seen_set]
        if new_items:
            train_dict.setdefault(u, []).extend(new_items)
            seen_set.update(new_items)
            seen_cache[u] = seen_set

        pred_dict[u] = [int(x) for x in topk_idx]

    return pred_dict


                    
      
                    
def main():
    parser = argparse.ArgumentParser(description="LightGCN Feedback Loop for Books (Pre-split)")
    parser.add_argument("--parts", type=int, default=PARTS, help="피드백 루프 스텝 수")
    parser.add_argument("--epochs", type=int, default=30, help="LightGCN 학습 epoch 수")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    
                                              
    print("Initializing Data from train.txt / label.txt ...")
    ground_truth, date2idx = get_initial_data_from_files()

                                  
    train_path = os.path.join(DATA_ROOT, TRAIN_REL)
    train_dict = _load_json(train_path)

                            
    all_timestamps = sorted({ts for interactions in ground_truth.values() for _, ts in interactions})
    if len(all_timestamps) == 0:
        raise RuntimeError("ground_truth의 timestamp가 비어 있습니다.")
    base_part_size = max(1, len(all_timestamps) // args.parts)
    predict_label = defaultdict(list)

                      
    for step in range(args.parts):
        print(f"\n=== Feedback Step {step+1}/{args.parts} ===")
        
                     
        train_dict = _load_json(train_path)
                                                                      
        for u in range(NUM_USERS_TOTAL):
            if u not in train_dict:
                train_dict[u] = []
                
            
        model, adj_t = train_lightgcn_softmax(
            train_dict=train_dict,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dropout=args.dropout,
            neg_k=10,
            num_items_total=NUM_ITEMS_TOTAL,               
            num_users_total=NUM_USERS_TOTAL
        )
        
                
        with torch.no_grad():
            user_emb, item_emb = model(adj_t)
        
            
        os.makedirs(RESULTS_DIR, exist_ok=True)
        np.save(os.path.join(RESULTS_DIR, f"user_emb_part{args.parts}_step{step}.npy"), user_emb.detach().cpu().numpy())
        np.save(os.path.join(RESULTS_DIR, f"item_emb_part{args.parts}_step{step}.npy"), item_emb.detach().cpu().numpy())

                        
        start_idx = step * base_part_size
        end_idx = (step + 1) * base_part_size if step < args.parts - 1 else len(all_timestamps)
        date_idx_range = set(all_timestamps[start_idx:end_idx])
        
        if not date_idx_range:
            print("[INFO] No dates in this window, skipping update.")
            continue

                   
        pred = recommend_and_update_by_datewindow(
            user_emb=user_emb,
            item_emb=item_emb,
            train_dict=train_dict,
            gt_by_dateidx=ground_truth,
            date_idx_range=date_idx_range,
        )

        for u, items in pred.items():
            if items:
                predict_label[u].extend(items)
        
                      
        _save_json(train_path, train_dict)
        
                      
        pred_path = os.path.join(RESULTS_DIR, f"predict_label_part{args.parts}_step{step}.json")
        with open(pred_path, "w") as f:
            json.dump({str(k): v for k, v in pred.items()}, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"[STEP {step}] Recommendations saved & train.json updated.")
    
              
    final_path = os.path.join(RESULTS_DIR, f"predict_label_part{args.parts}.json")
    with open(final_path, "w") as f:
        json.dump({str(k): v for k, v in predict_label.items()}, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"Final Loop Completed. Result -> {final_path}")

if __name__ == "__main__":
    main()