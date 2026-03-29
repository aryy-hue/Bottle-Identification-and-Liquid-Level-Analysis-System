from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import os
import time

def scrape_images_selenium(query, max_images, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    options = webdriver.ChromeOptions()
    
    options.binary_location = "/usr/sbin/chromium-browser"

    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    

    driver = webdriver.Chrome(options=options)
    
    search_url = f"https://www.bing.com/images/search?q={query}"
    driver.get(search_url)

    time.sleep(3)

    image_urls = set()
    skips = 0

    print(f"Searching for images of {query} on Bing...")

    while len(image_urls) < max_images:
        
        thumbnails = driver.find_elements(By.XPATH, "//img[contains(@class, 'mimg')]")

        for img in thumbnails[len(image_urls) + skips:]:
            try:
                src = img.get_attribute('src') or img.get_attribute('data-src')
                
                if src and 'http' in src:
                    image_urls.add(src)
                    print(f"Image Found: {len(image_urls)}")
                
                if len(image_urls) >= max_images:
                    break
            except Exception:
                continue
                
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(3) 

    driver.quit() 

    print(f"\nDownloading {len(image_urls)} images...")
    for i, url in enumerate(image_urls):
        try:
            headers = {'User-Agent': 'Mozilla/5.0'} 
            response = requests.get(url, headers=headers, timeout=10)
            
            file_path = os.path.join(folder_name, f"{query.replace(' ', '_')}_{i+1}.jpg")
            
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"[{i+1}/{max_images}] Image downloaded: {file_path}")
        except Exception as e:
            print(f"Failed to download image {i+1}. Error: {e}")

query = "medical bottle filled" 
max_images = 100
folder_name = "dataset_bottle"

scrape_images_selenium(query, max_images, folder_name)