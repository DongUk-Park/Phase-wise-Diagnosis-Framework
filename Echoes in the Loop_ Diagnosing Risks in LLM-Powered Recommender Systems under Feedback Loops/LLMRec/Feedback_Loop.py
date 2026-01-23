#!/usr/bin/env python3
        
from __future__ import annotations

import os
import sys
import json
import random
import argparse
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any, Optional

import numpy as np
import torch

import LLMRec
import data_construction
from LLM_augmentation_construct_prompt import main_generate
import LightGCN
     
                    
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ensure_list_int(v: Any) -> List[int]:
    if v is None:
        return []
    if isinstance(v, list):
        out = []
        for x in v:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out
    try:
        return [int(v)]
    except Exception:
        return []


def _load_json(path: str) -> Dict[str, List[int]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        raw = json.load(f)
    out: Dict[str, List[int]] = {}
    for k, v in raw.items():
        out[str(k)] = _ensure_list_int(v)
    return out


def _save_json(path: str, d: Dict[str, List[int]], *, indent: Optional[int] = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, ensure_ascii=False, indent=indent, sort_keys=True)


def _count_interactions_gt(ground_truth: Dict[int, List[Tuple[int, int]]]) -> int:
    return int(sum(len(v) for v in ground_truth.values()))


def _count_interactions_pred(pred: Dict[str, List[int]]) -> int:
    return int(sum(len(v) for v in pred.values()))


def _get_all_timestamps(ground_truth: Dict[int, List[Tuple[int, int]]]) -> List[int]:
    return sorted({ts for interactions in ground_truth.values() for _, ts in interactions})


def _active_users_in_time_range(
    ground_truth: Dict[int, List[Tuple[int, int]]],
    time_range: Set[int],
) -> List[int]:
    return [
        user for user, interactions in ground_truth.items()
        if any(ts in time_range for _, ts in interactions)
    ]


def _truncate_candidates_by_gt_count(
    best_candidates_full: Dict[int, List[int]],
    ground_truth: Dict[int, List[Tuple[int, int]]],
    active_users: List[int],
    time_range: Set[int],
) -> Dict[int, List[int]]:
       
    out: Dict[int, List[int]] = {}
    for u in active_users:
        if u not in best_candidates_full:
            continue
        k = len([1 for _, ts in ground_truth[u] if ts in time_range])
        out[u] = best_candidates_full[u][:k] if k > 0 else []
    return out


def _update_train_and_predict_json(
    json_path: str,
    predict_json_path: str,
    best_candidates_t: Dict[int, List[int]],
) -> None:
       
    train_dict = _load_json(json_path)
    predict_label = _load_json(predict_json_path)

    for user_id, items in best_candidates_t.items():
        user_id_str = str(user_id)

        if user_id_str not in train_dict:
            train_dict[user_id_str] = []
        train_dict[user_id_str].extend(items)

        if user_id_str not in predict_label:
            predict_label[user_id_str] = []
        predict_label[user_id_str].extend(items)

    _save_json(json_path, train_dict, indent=None)
    _save_json(predict_json_path, predict_label, indent=None)


                    
def main():
    parser = argparse.ArgumentParser(description="LLMRec Feedback Loop (refactored like traditionalCF script)")
    parser.add_argument("--file_path", type=str, default=DATA_ROOT)
    parser.add_argument("--folder", type=str, default=FOLDER)
    parser.add_argument("--dataset_name", type=str, default=DATASET_NAME)                             
    parser.add_argument("--parts", type=int, default=PARTS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    print("ENV CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    file_path = args.file_path
    folder = args.folder
    dataset_name = args.dataset_name
    parts = args.parts

                                 
    base_dir = os.path.join(file_path, folder)
    train_json_path = os.path.join(base_dir, "train.json")
    predict_json_path = os.path.join(base_dir, f"predict_label_part{parts}.json")

                                        
    ground_truth, date2idx = data_construction.get_data(file_path, folder)
    expected = _count_interactions_gt(ground_truth)
    print("expected ground_truth interactions:", expected)

                                                                              
    print("📁 Creating new predict_label.json")
    _save_json(predict_json_path, {}, indent=None)

                                             
    all_timestamps = _get_all_timestamps(ground_truth)
    if len(all_timestamps) == 0:
        raise RuntimeError("ground_truth timestamps is empty.")

    part_size = len(all_timestamps) // parts if parts > 0 else len(all_timestamps)

                                            
    for t in range(parts):
        print(f"\n🚀 Feedback Loop - Time Step t = {t}")

                                                             
        n_users, n_items = data_construction.get_train_matrix(file_path, folder)
        print(n_users, n_items)
        
                                    
        profiling_path = os.path.join(base_dir, "augmented_user_profiling_dict")
                
        if t >= 0:
            LightGCN.main(file_path, folder)

                                                                 
            n_tries = 2 if (t == 0 or t == parts - 1) else 1

            for n_try in range(n_tries):
                main_generate.main(dataset_name)

                augmented_user_profiling_dict = pickle.load(open(profiling_path, "rb"))
                pickle.dump(
                    augmented_user_profiling_dict,
                    open(
                        os.path.join(
                            base_dir,
                            f"augmented_user_profiling_dict_part{parts}_step{t}_try{n_try}",
                        ),
                        "wb",
                    ),
                )

                                    
        best_candidates_tensor, ua_embeddings, ia_embeddings = LLMRec.main()

        ua_np = ua_embeddings.detach().cpu().numpy()
        ia_np = ia_embeddings.detach().cpu().numpy()

        np.save(os.path.join(base_dir, f"user_emb_part{parts}_step{t}.npy"), ua_np)
        np.save(os.path.join(base_dir, f"item_emb_part{parts}_step{t}.npy"), ia_np)

                                                          
        best_candidates_full: Dict[int, List[int]] = {
            user_id: best_candidates_tensor[user_id].tolist()
            for user_id in range(best_candidates_tensor.shape[0])
        }

                                 
        start_idx = t * part_size
        end_idx = (t + 1) * part_size if t < parts - 1 else len(all_timestamps)
        time_range = set(all_timestamps[start_idx:end_idx])

                      
        active_users = _active_users_in_time_range(ground_truth, time_range)

                              
        active_set = set(active_users)
        cand_set = set(best_candidates_full.keys())
        missing = sorted(active_set - cand_set)
        print(f"[DEBUG] active_users={len(active_set)}, candidates_users={len(cand_set)}, missing_in_candidates={len(missing)}")
        if len(missing) > 0:
            print("[DEBUG] example missing users:", missing[:20])

                                            
        best_candidates_t = _truncate_candidates_by_gt_count(
            best_candidates_full=best_candidates_full,
            ground_truth=ground_truth,
            active_users=active_users,
            time_range=time_range,
        )
                                                                              
        _update_train_and_predict_json(
            json_path=train_json_path,
            predict_json_path=predict_json_path,
            best_candidates_t=best_candidates_t,
        )
                                                        
        max_time = max(ts for interactions in ground_truth.values() for _, ts in interactions)
        new_time = max_time + part_size
        print(f"✅ Updated train.json and predict_label.json for time {new_time}")

                                                       
    pred = _load_json(predict_json_path)
    actual = _count_interactions_pred(pred)
    print("actual predict_label interactions:", actual)
    print("diff:", expected - actual)


if __name__ == "__main__":
                       
    DATA_ROOT = "data/books"                                  
    FOLDER = "books_llmrec_format"                                                      
    DATASET_NAME = "books"                                                        
    PARTS = 5

    main()
