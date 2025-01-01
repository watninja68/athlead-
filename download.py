import csv
import re
from pytube import YouTube

# Path to the CSV file containing YouTube video URLs
csv_file_path = 'youtube_search_results.csv'

# Directory to save the downloaded videos
save_path = './vid'

def clean_url(url):
    # Use a regular expression to extract the video ID
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11}).*', url)
    if match:
        video_id = match.group(1)
        return f"https://www.youtube.com/watch?v={video_id}"
    return url

def download_video(url, save_path):
    try:
        yt = YouTube(url)
        stream = yt.streams.get_highest_resolution()
        stream.download(output_path=save_path)
        print(f"Downloaded: {yt.title}")
    except Exception as e:
        print(f"Failed to download {url}. Error: {str(e)}")

def get_video_urls(file_path):
    video_urls = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if 'url' in row:
                clean_video_url = clean_url(row['url'])
                video_urls.append(clean_video_url)
    return video_urls

# Get video URLs from the CSV file
video_urls = get_video_urls(csv_file_path)

# Download videos
for url in video_urls:
    download_video(url, save_path)