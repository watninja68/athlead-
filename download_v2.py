import csv
import yt_dlp as youtube_dl

csv_file_path = 'youtube_search_results.csv'

ydl_opts = {
    'format': 'best[ext=mp4][height<=720]',
    'outtmpl': '/vid/%(title)s.%(ext)s',  
}

with open(csv_file_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        title = row['title']
        url = row['url']
        print(f'Downloading: {title}')
        ydl_opts['outtmpl'] = f'/vid/{title}.mp4'
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
