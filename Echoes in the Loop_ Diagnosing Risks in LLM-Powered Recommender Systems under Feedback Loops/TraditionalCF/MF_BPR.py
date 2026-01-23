#!/usr/bin/env python3
                       

   

from __future__ import annotations
import os, json, math, random, argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    import pandas as pd                              
except Exception:
    pd = None

                              
DATA_ROOT = "data/ml-100k"
RAW_FILE  = "ml-100k_raw.txt"                                                           
SEED_OUT  = "feedback_loop/train.json"                                                     
PRED_OUT  = "feedback_loop/predict_label_for_feedback_loop.json"                               
NUM_ITEMS_TOTAL = 1682                                       
NUM_USERS_TOTAL = 943                                      

                       
       
                       
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

                       
                            
                       
def seed_feedback_from_raw(
    data_root: str = DATA_ROOT,
    raw_file: str = RAW_FILE,
    out_rel: str = SEED_OUT,
    take_lines: Optional[int] = None,                         
    force_init: bool = False,
):
       
    out_path = os.path.join(data_root, out_rel)
    raw_path = os.path.join(data_root, raw_file)

    need = force_init or (not os.path.exists(out_path))
    if not need:
        return

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"raw file not found: {raw_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

                              
    base = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                base = json.load(f)
        except Exception:
            base = {}

                                  
    if take_lines is None:
        total = 0
        with open(raw_path, "r") as f:
            for _ in f:
                total += 1
        take_lines = total // 2
        print(f"[INIT] Counting lines: total={total}, taking 50%={take_lines}")

                                        
    user2items = defaultdict(list)
    cnt = 0
    with open(raw_path, "r") as f:
        for line in f:
            if cnt >= take_lines:
                break
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u, i = int(parts[0]) - 1, int(parts[1]) - 1           
            if u < 0 or i < 0 or i >= NUM_ITEMS_TOTAL:
                continue
            user2items[u].append(i)
            cnt += 1

                
    for u, seq in user2items.items():
        k = str(u)
        if k not in base:
            base[k] = []
        base[k].extend(seq)

    with open(out_path, "w") as f:
        json.dump(base, f)
    print(f"[INIT] Seeded '{out_path}' from '{raw_path}' (first {take_lines} lines, 0-based).")

                       
                          
                       
def load_train_interactions_from_feedback(
    data_root: str = DATA_ROOT,
    rel: str = SEED_OUT,
    auto_seed_if_missing: bool = True,
) -> Dict[int, List[int]]:
       
    path = os.path.join(data_root, rel)
    if not os.path.exists(path):
        if auto_seed_if_missing:
            print(f"[WARN] train.json missing at '{path}'. Auto-seeding from raw (50%).")
            seed_feedback_from_raw(
                data_root=data_root, raw_file=RAW_FILE, out_rel=SEED_OUT,
                take_lines=None, force_init=False
            )
        else:
            raise FileNotFoundError(f"train.json not found: {path}")

    with open(path, "r") as f:
        raw = json.load(f)
    train = defaultdict(list)
    for k, v in raw.items():
        try:
            u = int(k)
        except Exception:
            continue
        if isinstance(v, list):
            for x in v:
                xi = int(x)
                if 0 <= xi < NUM_ITEMS_TOTAL:
                    train[u].append(xi)
        else:
            xi = int(v)
            if 0 <= xi < NUM_ITEMS_TOTAL:
                train[u].append(xi)
    return train

def load_interactions_from_raw(
    data_root: str = DATA_ROOT,
    raw_file: str = RAW_FILE,
    num_items_total: int = NUM_ITEMS_TOTAL,
) -> Dict[int, List[int]]:
       
    path = os.path.join(data_root, raw_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"raw file not found: {path}")
    user2items = defaultdict(list)
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            u, i = int(parts[0]) - 1, int(parts[1]) - 1
            if u < 0 or i < 0 or i >= num_items_total:
                continue
            user2items[u].append(i)
    return user2items


def split_train_test_by_ratio(user_dict: Dict[int, List[int]], train_ratio: float = 0.7):
                                        
    train_d, test_d = {}, {}
    for u, items in user_dict.items():
        n = len(items)
        if n == 0:
            train_d[u], test_d[u] = [], []; continue
        t = max(1, int(math.floor(n * train_ratio)))
        train_d[u], test_d[u] = items[:t], items[t:]
    return train_d, test_d

                       
                           
                       
class BPRDataset(Dataset):
                                       
    def __init__(self, train_dict: Dict[int, List[int]], num_items: int):
        super().__init__()
        self.num_items = num_items
        self.user_pos = {u: set(v) for u, v in train_dict.items()}
        self.users = list(self.user_pos.keys())
        self.all_items = set(range(num_items))
        self.pairs = [(u, i) for u, items in train_dict.items() for i in items if 0 <= i < num_items]
        self.user_neg_pool = {u: list(self.all_items - self.user_pos[u]) for u in self.users}

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        u, i = self.pairs[idx]
        pool = self.user_neg_pool[u]
        j = pool[random.randint(0, len(pool) - 1)] if pool else i
        return int(u), int(i), int(j)

def bpr_loss(u_ids, pos_i_ids, neg_i_ids, user_emb, item_emb, l2_reg: float = 0.0):
    pu = user_emb[u_ids]; qi = item_emb[pos_i_ids]; qj = item_emb[neg_i_ids]
    x_uij = (pu * qi).sum(1) - (pu * qj).sum(1)
    base = -F.logsigmoid(x_uij).mean()
    if l2_reg > 0:
        reg = (pu.pow(2).sum(1) + qi.pow(2).sum(1) + qj.pow(2).sum(1)).mean()
        return base + l2_reg * reg
    return base

                       
              
                       
class MF_BPR(nn.Module):
                                                         
    def __init__(self, num_users: int, num_items: int, embedding_dim: int):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, _adj_ignored=None) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.user_embedding.weight, self.item_embedding.weight

                       
              
                       
def train_mf_bpr(
    train_dict: Dict[int, List[int]],
    embedding_dim: int = 64,
    epochs: int = 30,
    batch_size: int = 2048,
    lr: float = 1e-3,
    l2_reg: float = 1e-4,
    num_items_total: int = NUM_ITEMS_TOTAL,
    device: Optional[str] = None,
    num_users_total: Optional[int] = NUM_USERS_TOTAL,                                    
    seed: int = 42,
) -> Tuple[MF_BPR, None]:
    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

                 
    if num_users_total is not None:
        num_users = int(num_users_total)
    else:
        max_u = max(train_dict.keys()) if len(train_dict) > 0 else 0
        num_users = int(max_u + 1)
    num_items = int(num_items_total)

    model = MF_BPR(num_users=num_users, num_items=num_items, embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ds = BPRDataset(train_dict, num_items=num_items)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    for ep in range(1, epochs + 1):
        model.train()
        total, steps = 0.0, 0
        for u, i, j in loader:
            u = u.to(device); i = i.to(device); j = j.to(device)
            user_emb, item_emb = model(None)
            loss = bpr_loss(u, i, j, user_emb, item_emb, l2_reg=l2_reg)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.item()); steps += 1
        print(f"[MF-BPR][Epoch {ep:02d}] loss={total/max(1,steps):.4f}")

    return model, None

                       
                                        
                       
@torch.no_grad()
def evaluate_recall_precision_hitratio_mf(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    train_seen: Dict[int, List[int]],
    test_dict: Dict[int, List[int]],
    Ks=(10, 20, 50)
):
    res = {K: {"recall": 0.0, "precision": 0.0, "hit": 0, "users": 0} for K in Ks}
    Kmax = max(Ks)
    device = user_emb.device

    for u, test_items in test_dict.items():
        if not test_items: continue
        if u < 0 or u >= user_emb.size(0):             
            continue

        scores = (user_emb[u:u+1] @ item_emb.T).squeeze(0)
        seen = train_seen.get(u, [])
        if seen:
            scores.index_fill_(0, torch.tensor(seen, dtype=torch.long, device=device), -1e9)
        topk = torch.topk(scores, k=min(Kmax, scores.numel())).indices.tolist()
        tset = set(test_items)
        for K in Ks:
            k_eff = min(K, len(topk))
            predK = set(topk[:k_eff])
            hits = len(predK & tset)
            res[K]["recall"] += hits / len(tset)
            res[K]["precision"] += hits / k_eff if k_eff > 0 else 0.0
            res[K]["hit"] += 1 if hits > 0 else 0
            res[K]["users"] += 1

    final = {}
    for K in Ks:
        n = res[K]["users"]
        final[K] = {"recall": 0.0, "precision": 0.0, "hitratio": 0.0, "users": n} if n == 0 else {
            "recall": res[K]["recall"]/n,
            "precision": res[K]["precision"]/n,
            "hitratio": res[K]["hit"]/n,
            "users": n
        }
    return final

@torch.no_grad()
def save_topk_predictions_mf(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    train_seen: Dict[int, List[int]],
    K: int,
    out_path: str,
):
    device = user_emb.device
    result = {}
    for u in train_seen.keys():
        if u < 0 or u >= user_emb.size(0):
            result[str(u)] = []; continue
        scores = (user_emb[u:u+1] @ item_emb.T).squeeze(0)
        seen = train_seen.get(u, [])
        if seen:
            scores.index_fill_(0, torch.tensor(seen, dtype=torch.long, device=device), -1e9)
        valid = (scores > -1e9/2).nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            result[str(u)] = []; continue
        k_eff = min(K, valid.numel())
        topk = torch.topk(scores, k=k_eff).indices.tolist()
        result[str(u)] = [int(i) for i in topk]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"[SAVE] Top-{K} predictions -> {out_path} (users={len(result)})")

                       
      
                       
def main():
    parser = argparse.ArgumentParser(description="MF-BPR (0-based + auto-seed 50% if missing)")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l2_reg", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_topk", action="store_true")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--use_global_user_count", action="store_true",
                        help="전체 데이터셋의 max user id + 1로 임베딩 크기 고정(콜드 유저 인덱싱 방지)")
    args = parser.parse_args()

    set_seed(args.seed)

                                            
                                                                                                            
    user2items_full = load_interactions_from_raw(DATA_ROOT, RAW_FILE, NUM_ITEMS_TOTAL)
    

                                                                      
    num_users_total_override = None
    if args.use_global_user_count:
        udata_path = os.path.join(DATA_ROOT, "u.data")
        if os.path.exists(udata_path):
            try:
                df = pd.read_csv(udata_path, sep="\t", header=None,
                                 names=["user_id","item_id","rating","timestamp"])
                global_u = int(df["user_id"].max())            
                num_users_total_override = global_u                                          
            except Exception:
                num_users_total_override = None

           
    train_dict, test_dict = split_train_test_by_ratio(user2items_full, train_ratio=args.train_ratio)

        
    model, _ = train_mf_bpr(
        train_dict=train_dict,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l2_reg=args.l2_reg,
        num_items_total=NUM_ITEMS_TOTAL,
        num_users_total=NUM_USERS_TOTAL,
        seed=args.seed,
    )

        
    with torch.no_grad():
        user_emb, item_emb = model(None)

    Ks = (10, 20, 50, 100)
    metrics = evaluate_recall_precision_hitratio_mf(
        user_emb=user_emb,
        item_emb=item_emb,
        train_seen=train_dict,
        test_dict=test_dict,
        Ks=Ks,
    )
    for K in Ks:
        m = metrics[K]
        print(f"[Eval@{K}] recall={m['recall']:.4f} | precision={m['precision']:.4f} | hitratio={m['hitratio']:.4f} | users={m['users']}")

                   
    if args.save_topk:
        out_pred = os.path.join(DATA_ROOT, PRED_OUT)
        save_topk_predictions_mf(
            user_emb=user_emb,
            item_emb=item_emb,
            train_seen=train_dict,
            K=args.K,
            out_path=out_pred,
        )

if __name__ == "__main__":
    main()
