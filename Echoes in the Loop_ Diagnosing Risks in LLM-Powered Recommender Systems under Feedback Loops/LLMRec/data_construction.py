import os
import json
import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle
from collections import defaultdict

def get_train(df):
        data = defaultdict(list)
        for user_id, group in df.groupby("user_id"):
            items = list(zip(group["item_id"], group["rating"]))
            for item in items:
                data[user_id].append(item[0])
        return data

def get_gt(test_path = None):
                  
    df = pd.read_csv(test_path, sep="\t", header=None, names=["user_id", "item_id", "timestamp"])
    
                        
    if df["timestamp"].dtype != object:
        df["date"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime('%Y-%m-%d')
    else:
        df["date"] = df["timestamp"]                  
                               
    unique_dates = sorted(df["date"].unique())
    date2idx = {date: idx for idx, date in enumerate(unique_dates)}

                                        
    data = defaultdict(list)
    for user_id, group in df.groupby("user_id"):
        items = list(zip(group["item_id"], group["date"]))
        for item in items:
            item_id, date = item
            date_idx = date2idx[date]         
            data[user_id].append((item_id, date_idx))
    
    return data, date2idx                  

def get_data(file_path="data/books", folder="books_llmrec_format"):
       
    
    test_path = os.path.join(file_path, "label.txt")
    train_path = os.path.join(file_path, "train.txt")
    train_data = {}
    with open(train_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u_str, i_str, _ = line.split()
            u = str(u_str)                  
            i = int(i_str)                                     

            if u not in train_data:
                train_data[u] = []
            train_data[u].append(i)

    test_data = {}
    with open(test_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u_str, i_str, _ = line.split()
            u = str(u_str)
            i = int(i_str)
                                                                       
            if u not in test_data:
                test_data[u] = []
            test_data[u].append(i)
            
                                  
    test_data_timestamp, date2idx = get_gt(test_path)

                           
    out_dir = f"{file_path}/{folder}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(os.path.join(out_dir, "train.json"), "w") as f:
        json.dump(train_data, f)
    with open(os.path.join(out_dir, "test.json"), "w") as f:
        json.dump(test_data, f)

    
    print("✅ Saved train.json, test.json")

    return test_data_timestamp, date2idx

def get_train_matrix(file_path = "data/books", folder="books_llmrec_format"):
                                             
    data_path = f"{file_path}/{folder}/"
    with open(os.path.join(data_path, "train.json"), "r") as f:
        train_dict = json.load(f)
    with open(os.path.join(data_path, "test.json"), "r") as f:
        test_dict = json.load(f)
                                        
    all_user_ids = set(map(int, train_dict.keys())) | set(map(int, test_dict.keys()))
    all_item_ids = set()
    for d in [train_dict, test_dict]:
        for items in d.values():
            all_item_ids.update(items)

    n_users = len(all_user_ids)
    n_items = len(all_item_ids)
    print(f"n_users={n_users}, n_items={n_items}")

                                       
    def build_sparse_matrix(data_dict, n_users, n_items):
        mat = sp.dok_matrix((n_users, n_items), dtype=np.float32)
        for user, items in data_dict.items():
            for item in items:
                if int(item) < n_items:
                    mat[int(user), int(item)] = 1.0
        return mat.tocsr()

    train_mat = build_sparse_matrix(train_dict, n_users, n_items)
    test_mat = build_sparse_matrix(test_dict, n_users, n_items)
    
                                           
    with open(os.path.join(data_path, "train_mat"), "wb") as f:
        pickle.dump(train_mat, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(data_path, "test_mat"), "wb") as f:
        pickle.dump(test_mat, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("✅ Saved train_mat, test_mat as .pkl files")
    return n_users, n_items
    
    
def main(data_path = "LLMRec/LLMRec_c/movielens"):
                          
    data_path = "data/ml-1m"
    folder = "ml-1m_llmrec_format"
    ground_truth, date2idx = get_data(data_path, folder)
    train_mat, test_mat = get_train_matrix(data_path, folder)
    
    print("finished")


if __name__ == "__main__":
    main()
    
