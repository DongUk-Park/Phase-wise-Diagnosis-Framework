import threading
import openai
import time
import pandas as pd
import pickle
import os
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm                                              

API_KEY = os.environ.get("OPENAI_API_KEY")

                          

def construct_prompting(item_attribute, indices, dataset):
                
    index = indices[0]                      
    
    if dataset.lower() == "netflix":
        year = item_attribute['year'][index]
        title = item_attribute['title'][index]
        pre_string = (
            "You are now a search engine, and required to provide the inquired information "
            "of the given movies below. Each movie includes year, title:\n"
        )
        item_list_string = f"[{index}] {year}, {title}\n"
        output_format = (
            "The inquired information is : director, country, language.\n"
            "Please output them in the following format:\n"
            "director::country::language\n"
            "please output only the content in the form above, i.e., director::country::language\n" 
            "Do not include reasoning, item index, or any extra text\n\n"
        )
    elif dataset.lower() == "movielens":
        title = item_attribute['title'][index]
        year = item_attribute['year'][index]
        genre = item_attribute['genre'][index]
        pre_string = (
            "You are now a search engine, and required to provide the inquired information "
            "of the given movies below. Each movie includes title, year, and genre:\n"
        )
        item_list_string = f"[{index}] {year}, {title}, {genre}\n"
        output_format = (
            "The inquired information is: director, country, language.\n"
            "Please output them in the following format:\n"
            "director::country::language\n"
            "Please output only the content in the format above, i.e., director::country::language.\n"
            "Do not include reasoning, item index, or any extra text.\n\n"
        )
    elif dataset.lower() == "books":
        brand = item_attribute['brand'][index]
        title = item_attribute['title'][index]
        category = item_attribute['category'][index]
        pre_string = (
            "You are now a search engine, and required to provide the inquired information "
            "of the given books below. Each book includes id, brand, title, and category:\n"
        )
        item_list_string = f"[{index}] {brand}, {title}, {category}\n"
        output_format = (
            "The inquired information is: author, country, language.\n"
            "Please output them in the following format:\n"
            "author::country::language\n"
            "Please output only the content in the format above, i.e., author::country::language.\n"
            "Do not include reasoning, item index, or any extra text.\n"
        )
    
    return pre_string + item_list_string + output_format


def LLM_request_worker(index, toy_item_attribute, model_type, dataset):
       
    try:
        indices = [index]
        prompt = construct_prompting(toy_item_attribute, indices, dataset)
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }
        params = {
            "model": model_type,
            "messages": [
                {"role": "system", "content": "You are now a search engine."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.6,
            "stream": False
        }

        response = requests.post(url=url, headers=headers, json=params, timeout=20)             

        if response.status_code != 200:
                                      
                                                                            
            return None

        message = response.json()
        if 'choices' not in message or 'message' not in message['choices'][0]:
            return None

        content = message['choices'][0]['message']['content']
                                                         

        rows = content.strip().split("\n")
                              
        if rows:
            elements = rows[0].split("::")
            if len(elements) == 3:
                director, country, language = elements
                return index, {
                    0: director.strip(),
                    1: country.strip(),
                    2: language.strip()
                }
    except Exception as e:
        print(f"❌ Exception for index {index}: {str(e)}")
        return None
    
    return None

def LLM_embedding_worker(index, row_data, keys, model_type):
       
    result_embeddings = {}
    
            
    MAX_RETRIES = 3
    BASE_TIMEOUT = 30                     
    
    for key in keys:
        try:
                                              
            text_input = str(row_data[key])
            if not text_input or text_input.lower() == 'nan':
                text_input = "unknown"

            url = "https://api.openai.com/v1/embeddings"
            headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            params = {
                "model": model_type,
                "input": text_input
            }
            
                            
            for attempt in range(MAX_RETRIES):
                try:
                    response = requests.post(
                        url=url, 
                        headers=headers, 
                        json=params, 
                        timeout=BASE_TIMEOUT          
                    )

                                    
                    if response.status_code == 200:
                        message = response.json()
                        if 'data' in message and message['data']:
                            result_embeddings[key] = message['data'][0]['embedding']
                        break                   
                    
                                                                  
                    elif response.status_code >= 500 or response.status_code == 429:
                        wait_time = 2 * (attempt + 1)                       
                        print(f"⚠️ Retry {index}-{key} (Code: {response.status_code}). Waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue        
                    
                                                           
                    else:
                        print(f"❌ Critical Error {index}-{key}: {response.status_code} - {response.text}")
                        break

                except requests.exceptions.Timeout:
                               
                    print(f"⏰ Timeout {index}-{key} (Attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(2)
                    continue
                
                except requests.exceptions.ConnectionError:
                                
                    print(f"🔌 Connection Error {index}-{key} (Attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(5)
                    continue
                    
                except Exception as e:
                    print(f"❌ Unknown Exception {index}-{key}: {str(e)}")
                    break

        except Exception as e:
            print(f"❌ Wrapper Exception {index}-{key}: {str(e)}")
            continue
            
                                                
    return index, result_embeddings

                    

def step1(file_path, model_type, error_cnt, dataset):
    print("step1 starts with Parallelization!")
    
    file_name = f"augmented_attribute_dict"
    full_path = os.path.join(file_path, file_name)

    if os.path.exists(full_path):
        print(f"✅ {file_name} exists. Loading...")
        with open(full_path, 'rb') as f:
            augmented_attribute_dict = pickle.load(f)
    else:
        print(f"❗ {file_name} does not exist. Initializing new file...")
        augmented_attribute_dict = {}

               
    if dataset.lower() == "netflix":
        df = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'year', 'title'])
    elif dataset.lower() == "movielens":
        df = pd.read_csv(os.path.join(file_path, 'item_attribute.csv'), names=['id', 'year', 'title', 'genre'])
    elif dataset.lower() == "books":
                           
        meta = pd.read_json("data/books/item_meta_2017_kcore10_user_item_split_filtered.json", lines=True)
        df = meta[["item_id", "brand", "title", "category"]].copy()
        df = df.rename(columns={"item_id": "id"})
        df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
        df.to_csv(os.path.join(file_path, 'item_attribute.csv'), index=False, header=None)
    else:
        raise ValueError(f"Unknown dataset type: {dataset}")

                                
    target_indices = [i for i in range(df.shape[0]) if i not in augmented_attribute_dict]
    print(f"Processing {len(target_indices)} items...")

              
    max_workers = 10                                            
    save_interval = 100            

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                      
        futures = {executor.submit(LLM_request_worker, idx, df, model_type, dataset): idx for idx in target_indices}
        
        completed_count = 0
        for future in tqdm(as_completed(futures), total=len(target_indices), desc="Step 1 Progress"):
            result = future.result()
            if result:
                idx, data = result
                augmented_attribute_dict[idx] = data
                completed_count += 1

                                  
            if completed_count % save_interval == 0:
                with open(full_path, 'wb') as f:
                    pickle.dump(augmented_attribute_dict, f)

           
    with open(full_path, 'wb') as f:
        pickle.dump(augmented_attribute_dict, f)

    print(f"\n✅ Step 1 completed: {file_name} updated.")


def step2(file_path, model_type, dataset):
                               
    print("step2 starts!")
    input_csv = os.path.join(file_path, 'item_attribute.csv')
    input_pkl = "augmented_attribute_dict"
    output_csv = os.path.join(file_path, "augmented_item_attribute_agg.csv")
    
    if dataset.lower() == "netflix":
        df = pd.read_csv(input_csv, names=['id', 'year', 'title'])
    elif dataset.lower() == "movielens":
        df = pd.read_csv(input_csv, names=['id', 'year', 'title', 'genre'])
    elif dataset.lower() == "books":
        df = pd.read_csv(input_csv, names=['id', 'brand', 'title', 'category'])

    with open(os.path.join(file_path, input_pkl), "rb") as f:
        attr_dict = pickle.load(f)
        
    if dataset.lower() == "movielens":
        director_list, country_list, language_list = [], [], []
        for i in range(len(df)):
            if i in attr_dict:
                director_list.append(attr_dict[i].get(0, 'unknown'))
                country_list.append(attr_dict[i].get(1, 'unknown'))
                language_list.append(attr_dict[i].get(2, 'unknown'))
            else:
                director_list.append("unknown")
                country_list.append("unknown")
                language_list.append("unknown")
        df['director'] = pd.Series(director_list)
        df['country'] = pd.Series(country_list)
        df['language'] = pd.Series(language_list)
        df.to_csv(output_csv, index=False, header=None)
        
    elif dataset.lower() == "books":
        author_list, country_list, language_list = [], [], []
        for i in range(len(df)):
            if i in attr_dict:
                author_list.append(attr_dict[i].get(0, 'unknown'))
                country_list.append(attr_dict[i].get(1, 'unknown'))
                language_list.append(attr_dict[i].get(2, 'unknown'))
            else:
                author_list.append("unknown")
                country_list.append("unknown")
                language_list.append("unknown")
        df['author'] = pd.Series(author_list)
        df['country'] = pd.Series(country_list)
        df['language'] = pd.Series(language_list)
        df.to_csv(output_csv, index=False, header=None)
    
                                 

    print(f"\n✅ Step 2 completed: Aggregated CSV saved to {output_csv}")


def step3(file_path, model_type, emb_model, dataset):
       
    print("step3 starts with Parallelization!")
    
    batch_size = 500
    max_workers = 10                                 

               
    if dataset.lower() == "movielens":
        df = pd.read_csv(file_path + '/augmented_item_attribute_agg.csv', names=["id", "year", "title", "genre", "director", "country", "language"])
    elif dataset.lower() == "books":
        df = pd.read_csv(file_path + '/augmented_item_attribute_agg.csv', names=["id", "brand", "title", "category", "author", "country", "language"])    
    
    cols = [col for col in df.columns if col != 'id']
    for col in cols:
        df[col] = df[col].fillna("unknown").astype(str)

    total_items = df.shape[0]
    num_batches = (total_items + batch_size - 1) // batch_size

    for dict_idx in range(1, num_batches + 1):
        file_name = f"augmented_attribute_embedding_dict{dict_idx}"
        full_path = os.path.join(file_path, file_name)

        if os.path.exists(full_path):
            print(f"✅ Skipping {file_name} (already exists)")
            continue
        
                         
        augmented_attribute_embedding_dict = {col: {} for col in cols}

        start_index = (dict_idx - 1) * batch_size
        end_index = min(start_index + batch_size, total_items)
        
        print(f"Processing Batch {dict_idx}/{num_batches} ({start_index}~{end_index})...")

                             
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for i in range(start_index, end_index):
                                       
                row_data = df.iloc[i].to_dict()
                futures.append(executor.submit(LLM_embedding_worker, i, row_data, cols, emb_model))
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    idx, embeddings = result
                                          
                    for col, emb in embeddings.items():
                        augmented_attribute_embedding_dict[col][idx] = emb

        with open(full_path, 'wb') as f:
            pickle.dump(augmented_attribute_embedding_dict, f)
        print(f"✅ Saved: {file_name}")

def step4(file_path, dataset):
                
    print("step4 starts!")
    
    if dataset.lower() == "movielens":
        total_dict = {'year': {}, 'title': {}, 'genre':{}, 'director': {}, 'country': {}, 'language': {}}
    elif dataset.lower() == "books":
        total_dict = {'brand': {}, 'title': {}, 'category':{}, 'author': {}, 'country': {}, 'language': {}}

    i = 1
    while True:
        file_name = f"augmented_attribute_embedding_dict{i}"
        full_path = os.path.join(file_path, file_name)
        if not os.path.exists(full_path):
            break
        with open(full_path, 'rb') as f:
            tmp_dict = pickle.load(f)
        for key in total_dict.keys():
            total_dict[key].update(tmp_dict[key])
        print(f"✅ Aggregated {file_name}")
        i += 1

    with open(file_path + '/augmented_attribute_embedding_dict', 'wb') as f:
        pickle.dump(total_dict, f)
    print(f"\n✅ Aggregated dict saved to: augmented_attribute_embedding_dict\n")

def step5(file_path):
                
    print("step5 starts!")
    
    with open(file_path + '/augmented_attribute_embedding_dict', "rb") as f:
        aggregated_dict = pickle.load(f)

    total_matrix = {}
    
                        
    try:
        with open(file_path + '/train_mat', 'rb') as f:
            train_mat = pickle.load(f)
        n_items = train_mat.shape[1]
    except FileNotFoundError:
        print("Warning: train_mat not found. Using max index from aggregated_dict.")
        n_items = 0
        for k in aggregated_dict:
            if aggregated_dict[k]:
                n_items = max(n_items, max(aggregated_dict[k].keys()) + 1)

    for key in aggregated_dict:
        value_dict = aggregated_dict[key]
        vectors = []
        for i in range(n_items):
            if i in value_dict:
                vectors.append(value_dict[i])
            else:
                vectors.append(np.zeros(1536))              
        total_matrix[key] = np.array(vectors)
        print(f"{key} embedding shape: {total_matrix[key].shape}")

    with open(file_path + '/augmented_total_embed_dict', 'wb') as f:
        pickle.dump(total_matrix, f)

    print(f"\n✅ Numpy embedding matrix saved to: augmented_total_embed_dict\n")

def step6(file_path, key='title', top_k=10):
                
    print("step6 starts!")
    
    with open(file_path + '/augmented_total_embed_dict', 'rb') as f:
        embed_dict = pickle.load(f)

    if key not in embed_dict:
        raise ValueError(f"'{key}' not found in augmented_total_embed_dict")

    emb_matrix = embed_dict[key]
    sim_matrix = cosine_similarity(emb_matrix)

    num_items = sim_matrix.shape[0]
    edge_list = []
    for i in range(num_items):
                                       
        top_indices = np.argsort(sim_matrix[i])[::-1][1:top_k+1] 
        for j in top_indices:
            edge_list.append((i, j, sim_matrix[i][j]))

    df_edges = pd.DataFrame(edge_list, columns=['source', 'target', 'weight'])
    df_edges.to_csv(os.path.join(file_path, f"{key}_similarity_edges.csv"), index=False)
    print(f"\n✅ {key} similarity-based i-i edge file saved: {key}_similarity_edges.csv\n")


def main():
    openai.api_key = API_KEY
    model_type = "gpt-4o" 
    emb_model= "text-embedding-3-small"
    
    dataset = "movielens"                                   
                           
    if dataset == "netflix":
        file_path = "LLMRec/LLMRec_c/" + dataset + "/netflix_valid_item"
    elif dataset == "movielens":
        file_path = "data/ml-1m/ml-1m_llmrec_format/"
    elif dataset == "books":
        file_path = "data/books/books_llmrec_format/"

    error_cnt=0
    
                  
    step1(file_path, model_type, error_cnt, dataset) 
    step2(file_path, model_type, dataset)
    step3(file_path, model_type, emb_model, dataset)
    step4(file_path, dataset)
    step5(file_path)
                                             

if __name__ == '__main__':
    main()