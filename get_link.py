from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import csv

chrome_driver_path = 'C:/Users/Home/Documents/chrome drivers/chromedriver-win64/chromedriver.exe'
options = Options()
service = Service(executable_path=chrome_driver_path)
driver = webdriver.Chrome(service=service, options=options)

def search_youtube(query, num_results=20):
    try:
        driver.get("https://www.youtube.com")
        try:
            cookie_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[@aria-label='Accept all']"))
            )
            cookie_button.click()
        except:
            pass
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.NAME, "search_query"))
        )
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.ID, "video-title"))
        )
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        while len(driver.find_elements(By.ID, "video-title")) < num_results:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(2)
            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        video_links = []
        for element in driver.find_elements(By.ID, "video-title")[:num_results]:
            link = element.get_attribute("href")
            title = element.text
            if link and title:
                video_links.append({"title": title, "url": link})
        return video_links
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def save_to_csv(video_links, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['title', 'url']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for video in video_links:
            writer.writerow(video)
    print(f"Results saved to {filename}")

query = "track and field running"
results = search_youtube(query, num_results=200)
save_to_csv(results, "youtube_search_results.csv")
for video in results:
    print(f"Title: {video['title']}")
    print(f"URL: {video['url']}")
    print('---')
driver.quit()
