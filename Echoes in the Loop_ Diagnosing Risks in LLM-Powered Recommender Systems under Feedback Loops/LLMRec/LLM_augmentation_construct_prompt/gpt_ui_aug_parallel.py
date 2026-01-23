import threading
import os
import time
import pickle
import requests
import pandas as pd
import numpy as np
import concurrent.futures
from tqdm import tqdm
import torch


API_KEY = os.environ.get("OPENAI_API_KEY")

                                        
def construct_prompting(item_attribute, item_list, candidate_list, dataset):
    if dataset.lower() == "netflix":
        history_string = "User history:\n"
        for index in item_list:
            year = item_attribute['year'][index]
            title = item_attribute['title'][index]
            history_string += f"[{index}] {year}, {title}\n"

        candidate_string = "Candidates:\n"
        for index in candidate_list:
            idx = index.item() if isinstance(index, (torch.Tensor, np.generic)) else int(index)
            year = item_attribute['year'][idx]
            title = item_attribute['title'][idx]
            candidate_string += f"[{idx}] {year}, {title}\n"
        
        output_format = (
            "Please output the index of user's favorite and least favorite movie only from candidate, but not user history.\n"
            "Output format:\nTwo numbers separated by '::'. Nothing else.\n"
            "Please just give the index of candidates, remove [], do not output other things, no reasoning.\n\n"
        )
        
        prompt = (
            "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title (same topic/doctor), year (similar years), genre (similar genre).\n"
            + history_string + candidate_string + output_format
        )

    elif dataset.lower() == "movielens":
        history_string = "User history:\n"
        for index in item_list:
            title = item_attribute['title'][index]
            year = item_attribute['year'][index]
            genre = item_attribute['genre'][index]
            history_string += f"[{index}] {year}, {title}, {genre}\n"

        candidate_string = "Candidates:\n"
        for index in candidate_list:
            idx = index.item() if isinstance(index, (torch.Tensor, np.generic)) else int(index)
            title = item_attribute['title'][idx]
            year = item_attribute['year'][idx]
            genre = item_attribute['genre'][idx]
            candidate_string += f"[{idx}] {year}, {title}, {genre}\n"

        output_format = (
            "Please output the index of user's favorite and least favorite movie only from candidate, but not user history.\n"
            "Output format:\nTwo numbers separated by '::'. Nothing else.\n"
            "Please just give the index of candidates, remove [], do not output other things, no reasoning.\n\n"
        )
        
        prompt = (
            "You are a movie recommendation system and required to recommend user with movies based on user history that each movie with title (same topic/doctor), year (similar years), genre (similar genre).\n"
            + history_string + candidate_string + output_format
        )

    elif dataset.lower() == "books":
        history_string = "User history:\n"
        for index in item_list:
            brand = item_attribute['brand'][index]
            title = item_attribute['title'][index]
            category = item_attribute['category'][index]
            history_string += f"[{index}] {brand}, {title}, {category}\n"

        candidate_string = "Candidates:\n"
        for index in candidate_list:
            idx = index.item() if isinstance(index, (torch.Tensor, np.generic)) else int(index)
            brand = item_attribute['brand'][idx]
            title = item_attribute['title'][idx]
            category = item_attribute['category'][idx]
            candidate_string += f"[{idx}] {brand}, {title}, {category}\n"
        
        output_format = (
            "Please output the index of user's favorite and least favorite book only from candidate, but not user history.\n"
            "Output format:\nTwo numbers separated by '::'. Nothing else.\n"
            "Please just give the index of candidates, remove [], do not output other things, no reasoning.\n\n"
        )   
        prompt = (
            "You are a book recommendation system and required to recommend user with books based on user history that each book with brand, title, category.\n"
            + history_string + candidate_string + output_format
        )
    
    return prompt

                                       
def LLM_request_worker(args):
    index, toy_item_attribute, adjacency_list, candidate_list, model_type, dataset = args

    prompt = construct_prompting(toy_item_attribute, adjacency_list, candidate_list, dataset)
    
    if dataset.lower() == "books":
        sys_msg = "You are a book recommendation system."
    else:
        sys_msg = "You are a movie recommendation system."

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + API_KEY
    }
    params = {
        "model": model_type,
        "messages": [
            {"role": "system", "content": sys_msg}, 
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,              
        "temperature": 0.7,
        "stream": False
    }

    for retry in range(3):                
        try:
            response = requests.post(url=url, headers=headers, json=params, timeout=20)
            
            if response.status_code != 200:
                                                                                
                time.sleep(2)
                continue

            message = response.json()
            content = message['choices'][0]['message']['content']
            
                   
            samples = content.strip().split("::")
            if len(samples) < 2:
                                             
                import re
                nums = re.findall(r'\d+', content)
                if len(nums) >= 2:
                    pos_sample = int(nums[0])
                    neg_sample = int(nums[1])
                else:
                    raise ValueError("Parsing Failed")
            else:
                pos_sample = int(samples[0].strip())
                neg_sample = int(samples[1].strip())

                                                                               
            return index, {0: pos_sample, 1: neg_sample}

        except Exception as e:
                                                     
            time.sleep(2)

    return index, None               

def main(dataset="books"):
           
    if dataset == "netflix":
        file_path = "LLMRec/LLMRec_c/" + dataset + "/netflix_valid_item/"
    elif dataset == "movielens":
        file_path = "data/ml-1m/ml-1m_llmrec_format/"
    elif dataset == "books":
        file_path = "data/books/books_llmrec_format/"
    
    model_type = "gpt-4o"
    aug_path = os.path.join(file_path, "augmented_sample_dict")

               
    print("📂 Loading Data...")
    candidate_indices = pickle.load(open(os.path.join(file_path, 'candidate_indices'), 'rb'))
    candidate_indices_dict = {i: candidate_indices[i] for i in range(candidate_indices.shape[0])}

    train_mat = pickle.load(open(os.path.join(file_path, 'train_mat'),'rb'))
    adjacency_list_dict = {}
    for index in range(train_mat.shape[0]):
        _, data_y = train_mat[index].nonzero()
        adjacency_list_dict[index] = data_y
    
    if dataset == "netflix":
        toy_item_attribute = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'year', 'title'])
    elif dataset == "movielens":
        toy_item_attribute = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'year', 'title', 'genre'])
    elif dataset == "books":
        toy_item_attribute = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'brand', 'title', 'category'])

                    
    augmented_sample_dict = {}
    print("🆕 Starting new dictionary.")

                                 
    all_indices = list(adjacency_list_dict.keys())
    target_indices = [i for i in all_indices if i not in augmented_sample_dict]
    
    print(f"🚀 Processing {len(target_indices)} users with Multithreading...")

    failed_indices = []
    max_workers = 10            
    save_interval = 20        
    batch_cnt = 0

                 
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                 
        futures = {
            executor.submit(
                LLM_request_worker, 
                (idx, toy_item_attribute, adjacency_list_dict[idx][-10:], candidate_indices_dict[idx], model_type, dataset)
            ): idx for idx in target_indices
        }

                               
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(target_indices), desc="Augmenting"):
            idx, result = future.result()
            
            if result is not None:
                augmented_sample_dict[idx] = result
                batch_cnt += 1
            else:
                failed_indices.append(idx)

                   
            if batch_cnt >= save_interval:
                with open(aug_path, 'wb') as f:
                    pickle.dump(augmented_sample_dict, f)
                batch_cnt = 0
                                            

           
    with open(aug_path, 'wb') as f:
        pickle.dump(augmented_sample_dict, f)
    
              
    if failed_indices:
        with open(os.path.join(file_path, 'failed_ui_aug_indices.pkl'), 'wb') as f:
            pickle.dump(failed_indices, f)
        print(f"❗ {len(failed_indices)} indices failed.")
    else:
        print("✅ All processed successfully.")

if __name__ == '__main__':
                                             
    main("books")