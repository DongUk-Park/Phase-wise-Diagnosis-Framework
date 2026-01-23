#!/usr/bin/env python3
                       

import os
import gzip
import json
import pickle
import ast
from tqdm import tqdm


def _safe_listify_desc(x):
       
    if x is None:
        return ""
            
    if isinstance(x, list):
        return " ".join(str(s) for s in x if s)
             
    s = str(x)
    s_stripped = s.strip()
                          
    if s_stripped.startswith("[") and s_stripped.endswith("]"):
        try:
            parsed = ast.literal_eval(s_stripped)
            if isinstance(parsed, list):
                return " ".join(str(t) for t in parsed if t)
        except Exception:
            pass
    return s

def _safe_join_category(cat):
       
    if cat is None:
        return ""
    if isinstance(cat, list):
        return ", ".join(str(c) for c in cat if c)
    return str(cat)

def build_books_name_dict(meta_path: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    name_dict = {"title": {}, "description": {}}

                   
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading meta JSONL"):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

                      
            item_id = obj.get("item_id", None)
            title   = obj.get("title", "")
            brand   = obj.get("brand", "")
            category = obj.get("category", [])
            desc    = obj.get("description", "")

            if item_id is None:
                continue                      

                       
            try:
                item_id = int(item_id)
            except Exception:
                continue

                 
            title_str = "" if title is None else str(title).strip()
            brand_str = "" if brand is None else str(brand).strip()
            cat_str   = _safe_join_category(category).strip()
            desc_str  = _safe_listify_desc(desc).strip()

            custom_desc = f"[{brand_str}], [{cat_str}], [{desc_str}]"

            name_dict["title"][item_id] = title_str
            name_dict["description"][item_id] = custom_desc
                               
                       
    
    with gzip.open(out_path, "wb") as tf:
        pickle.dump(name_dict, tf)
    print(f"✅ Saved name_dict: {out_path} (items: {len(name_dict['title'])})")

if __name__ == "__main__":
    META_PATH = "data/books/item_meta_2017_kcore10_user_item.json"         
    OUT_DIR   = "A-LLMRec/data/books"
    OUT_PATH  = os.path.join(OUT_DIR, "books_text_name_dict.json.gz")
    build_books_name_dict(META_PATH, OUT_PATH)
