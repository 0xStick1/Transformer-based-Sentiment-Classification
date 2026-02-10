import requests
import os
import time

def download_file(url, filename, proxies=None):
    print(f"Attempting to download from {url}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://hf-mirror.com/'
    }
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=60, proxies=proxies)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and downloaded % (1024*1024) == 0:
                        print(f"Downloaded {downloaded/1024/1024:.2f} MB / {total_size/1024/1024:.2f} MB")
        
        # Verify file size (should be > 1MB)
        if os.path.exists(filename) and os.path.getsize(filename) > 1024 * 1024:
            print(f"File downloaded successfully to {filename}")
            return True
        else:
            print("Download completed but file seems too small. Likely failed.")
            return False
            
    except Exception as e:
        print(f"Failed to download from {url}. Error: {e}")
        return False

def main():
    filename = "online_shopping_10_cats.csv"
    if os.path.exists(filename) and os.path.getsize(filename) > 1024 * 1024:
        print(f"{filename} already exists and looks valid. Skipping download.")
        return

    # Prioritize Hugging Face Mirror which is usually faster/more stable in CN
    # This URL points to the raw file in a repo that hosts this dataset
    urls = [
        "https://hf-mirror.com/datasets/XiangPan/online_shopping_10_cats_62k/resolve/main/online_shopping_10_cats.csv",
        "https://mirror.ghproxy.com/https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/online_shopping_10_cats/online_shopping_10_cats.csv",
    ]
    
    for url in urls:
        if download_file(url, filename):
            return
        time.sleep(2) # Wait a bit before next attempt
        
    print("All download attempts failed. Please manually download online_shopping_10_cats.csv")

if __name__ == "__main__":
    main()
