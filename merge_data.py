import pandas as pd
import requests
import os
import time

def download_file(url, filename):
    if os.path.exists(filename) and os.path.getsize(filename) > 1024 * 1024:
        print(f"Skipping download, {filename} already exists.")
        return True
    
    print(f"Downloading {filename} from {url}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return False

def merge_datasets():
    data_configs = [
        {
            'name': 'online_shopping_10_cats.csv',
            'url': 'https://hf-mirror.com/datasets/XiangPan/online_shopping_10_cats_62k/resolve/main/online_shopping_10_cats.csv',
            'text_col': 'review',
            'label_col': 'label'
        },
        {
            'name': 'waimai_10k.csv',
            'url': 'https://hf-mirror.com/datasets/dirtycomputer/waimai_10k/resolve/main/waimai_10k.csv',
            'text_col': 'review',
            'label_col': 'label'
        },
        {
            'name': 'weibo_senti_100k.csv',
            'url': 'https://hf-mirror.com/datasets/dirtycomputer/weibo_senti_100k/resolve/main/weibo_senti_100k.csv',
            'text_col': 'review',
            'label_col': 'label'
        }
    ]

    all_dfs = []
    
    for config in data_configs:
        if download_file(config['url'], config['name']):
            try:
                df = pd.read_csv(config['name'])
                # Standardize to label and review columns
                df = df[[config['label_col'], config['text_col']]]
                df.columns = ['label', 'review']
                all_dfs.append(df)
                print(f"Loaded {len(df)} samples from {config['name']}")
            except Exception as e:
                print(f"Error processing {config['name']}: {e}")
        time.sleep(1)

    if not all_dfs:
        print("No datasets were loaded.")
        return

    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df = merged_df.dropna(subset=['review'])
    merged_df = merged_df.drop_duplicates(subset=['review'])
    
    output_name = 'merged_sentiment_data.csv'
    merged_df.to_csv(output_name, index=False)
    
    print("\n" + "="*30)
    print(f"Merge Complete!")
    print(f"Total samples: {len(merged_df)}")
    print(f"Saved to: {output_name}")
    print("="*30)

if __name__ == "__main__":
    merge_datasets()
