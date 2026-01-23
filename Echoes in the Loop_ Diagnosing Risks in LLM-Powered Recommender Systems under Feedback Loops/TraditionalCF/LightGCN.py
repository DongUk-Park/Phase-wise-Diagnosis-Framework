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

                              
DATA_ROOT = "data/books"
TRAIN_RAW = "train.txt"
TRAIN_REL  = "traditionalCF/train.json"
                  
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)   
                       
class LightGCN(nn.Module):
    def __init__(self, num_users: int, num_items: int, embedding_dim: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.num_users, self.num_items = num_users, num_items
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        self.num_layers, self.dropout = num_layers, dropout

    def forward(self, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert adj.is_sparse, "adj must be sparse"
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], 0)
        embs = [all_emb]
        for _ in range(self.num_layers):
            all_emb = torch.sparse.mm(adj, all_emb)
            if self.dropout > 0:
                all_emb = F.dropout(all_emb, p=self.dropout, training=self.training)
            embs.append(all_emb)
        final_emb = torch.mean(torch.stack(embs, 1), 1)
        return final_emb[:self.num_users], final_emb[self.num_users:]

                  
def load_train_interactions_from_json(path: str) -> Dict[int, List[int]]:
                                  
    if not os.path.exists(path):
        raise FileNotFoundError(f"Train JSON not found: {path}")
    
    with open(path, "r") as f:
        raw = json.load(f)
    
    train = defaultdict(list)
    for k, v in raw.items():
        try:
            u = int(k)
        except: continue
        if isinstance(v, list):
            train[u] = [int(x) for x in v]
        else:
            train[u] = [int(v)]
    return train

def build_norm_adj(train_dict: Dict[int, List[int]], num_users: int, num_items: int) -> torch.Tensor:
    rows, cols = [], []
    for u, items in train_dict.items():
        for i in set(items):
                                             
            if i >= num_items: continue 
            ui, vi = u, num_users + i
            rows += [ui, vi]; cols += [vi, ui]
    N = num_users + num_items
    
    if not rows:
        idx = torch.empty((2, 0), dtype=torch.long)
        val = torch.empty((0,), dtype=torch.float32)
        return torch.sparse_coo_tensor(idx, val, (N, N)).coalesce()
        
    row = torch.tensor(rows, dtype=torch.long)
    col = torch.tensor(cols, dtype=torch.long)
    
    deg = torch.bincount(row, minlength=N).to(torch.float32)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    
    values = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    indices = torch.stack([row, col], 0)
    return torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()

                       
class BPRDataset(Dataset):
    def __init__(self, train_dict: Dict[int, List[int]], num_items: int):
        super().__init__()
        self.user_pos = {u: set(v) for u, v in train_dict.items() if v}              
        self.users = list(self.user_pos.keys())
        self.num_items = num_items
        self.pairs = []
        for u, items in train_dict.items():
            for i in items:
                self.pairs.append((u, i))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        u, i = self.pairs[idx]
                           
        while True:
            j = random.randint(0, self.num_items - 1)
            if j not in self.user_pos[u]:
                break
        return int(u), int(i), int(j)

class SoftmaxNegKDataset(Dataset):
       
    def __init__(self, train_dict: Dict[int, List[int]], num_items: int, neg_k: int = 10):
        super().__init__()
        self.neg_k = neg_k
        self.num_items = num_items

                            
        self.user_pos = {u: set(v) for u, v in train_dict.items() if v}
        self.users = list(self.user_pos.keys())

                                       
        all_items = set(range(num_items))
        self.user_neg_pool = {u: list(all_items - pos) for u, pos in self.user_pos.items()}

                           
        self.pairs: List[Tuple[int, int]] = []
        for u, items in train_dict.items():
            if not items:
                continue
            for i in items:
                if 0 <= i < num_items:
                    self.pairs.append((int(u), int(i)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        u, pos = self.pairs[idx]
        neg_pool = self.user_neg_pool.get(u, [])
        if len(neg_pool) == 0:
            negs = [pos] * self.neg_k
        else:
                                                                      
            negs = [neg_pool[random.randint(0, len(neg_pool)-1)] for _ in range(self.neg_k)]
        return int(u), int(pos), torch.tensor(negs, dtype=torch.long)


def bpr_loss(u_ids, pos_i_ids, neg_i_ids, user_emb, item_emb):
    u = user_emb[u_ids]
    pos = item_emb[pos_i_ids]
    neg = item_emb[neg_i_ids]
    x = (u * pos).sum(1) - (u * neg).sum(1)
    return -F.logsigmoid(x).mean()

def sampled_softmax_loss(u_ids, pos_i_ids, neg_i_ids, user_emb, item_emb):
       
    u = user_emb[u_ids]                             
    pos = item_emb[pos_i_ids]                       
    neg = item_emb[neg_i_ids]                          

    pos_logit = (u * pos).sum(dim=1, keepdim=True)             
    neg_logits = torch.einsum('bd,bkd->bk', u, neg)            
    logits = torch.cat([pos_logit, neg_logits], dim=1)           

    targets = torch.zeros(u.shape[0], dtype=torch.long, device=u_ids.device)           
    return F.cross_entropy(logits, targets)

              
def train_lightgcn(train_dict, embedding_dim=64, num_layers=2, epochs=50, batch_size=2048, 
                   lr=1e-3, dropout=0.0, num_items_total=None, num_users_total=None):
    
                                
    if num_users_total is None:
        num_users_total = max(train_dict.keys()) + 1 if train_dict else 0
    if num_items_total is None:
                                                       
        max_item = 0
        for items in train_dict.values():
            if items: max_item = max(max_item, max(items))
        num_items_total = max_item + 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training LightGCN on {device} | Users: {num_users_total}, Items: {num_items_total}")
    
    model = LightGCN(num_users_total, num_items_total, embedding_dim, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    adj_t = build_norm_adj(train_dict, num_users_total, num_items_total).to(device)
    dataset = BPRDataset(train_dict, num_items_total)
    
    if len(dataset) == 0:
        print("Warning: Dataset is empty.")
        return model, adj_t

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False,
                        num_workers=4, pin_memory=True)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        for u, i, j in loader:
            u, i, j = u.to(device), i.to(device), j.to(device)
            user_emb, item_emb = model(adj_t)
            loss = bpr_loss(u, i, j, user_emb, item_emb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            steps += 1
        
        if ep % 10 == 0:
            avg = total_loss / steps if steps > 0 else 0
            print(f"[Epoch {ep:02d}] Loss: {avg:.4f}")

    return model, adj_t

def train_lightgcn_softmax(
    train_dict,
    embedding_dim=64,
    num_layers=2,
    epochs=50,
    batch_size=2048,
    lr=1e-3,
    dropout=0.0,
    neg_k=10,
    num_items_total=None,
    num_users_total=None,
):
                                
    if num_users_total is None:
        num_users_total = max(train_dict.keys()) + 1 if train_dict else 0
    if num_items_total is None:
        max_item = 0
        for items in train_dict.values():
            if items:
                max_item = max(max_item, max(items))
        num_items_total = max_item + 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training LightGCN(SoftmaxNegK) on {device} | Users: {num_users_total}, Items: {num_items_total}")

    model = LightGCN(num_users_total, num_items_total, embedding_dim, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    adj_t = build_norm_adj(train_dict, num_users_total, num_items_total).to(device)

    dataset = SoftmaxNegKDataset(train_dict, num_items_total, neg_k=neg_k)
    if len(dataset) == 0:
        print("Warning: Dataset is empty.")
        return model, adj_t

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )

    for ep in range(1, epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0

        for u, pos, negs in loader:
            u = u.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            negs = negs.to(device, non_blocking=True)          

            user_emb, item_emb = model(adj_t)

            loss = sampled_softmax_loss(u, pos, negs, user_emb, item_emb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        if ep % 10 == 0 or ep == 1:
            avg = total_loss / max(1, steps)
            print(f"[Epoch {ep:02d}] Loss: {avg:.4f}")

    return model, adj_t

                       
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

                                          
    raw_path = os.path.join(DATA_ROOT, TRAIN_RAW)
    if os.path.exists(raw_path):
        import pandas as pd
        print(f"Loading from {raw_path}...")
        df = pd.read_csv(raw_path, sep=None, engine='python', names=['u', 'i', 'ts'])
        train_dict = {u: list(g['i']) for u, g in df.groupby('u')}
        
                      
        num_users = df['u'].max() + 1
        num_items = df['i'].max() + 1
        
        train_lightgcn_softmax(train_dict, epochs=args.epochs, 
                       num_items_total=num_items, num_users_total=num_users)
    else:
        print(f"File not found: {raw_path}")

if __name__ == "__main__":
    main()