from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from urllib.parse import urlparse, parse_qs

def extract_video_id(url: str) -> str:
    """
    Extracts the YouTube video ID from any valid YouTube URL.
    Supports formats:
      - https://www.youtube.com/watch?v=xxxx
      - https://youtu.be/xxxx
      - https://www.youtube.com/embed/xxxx
    """
    parsed = urlparse(url)

    # Case 1: Standard watch URL
    if parsed.hostname in ["www.youtube.com", "youtube.com"]:
        query = parse_qs(parsed.query)
        if "v" in query:
            return query["v"][0]

        # Case 2: Embed URLs
        if parsed.path.startswith("/embed/"):
            return parsed.path.split("/embed/")[1]

    # Case 3: Short URLs (youtu.be)
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")

    raise ValueError("Invalid YouTube URL or cannot extract video ID")

url = "https://www.youtube.com/watch?v=pPRoAs8xh2o"  

try:
    video_id = extract_video_id(url)
    print("Extracted video ID:", video_id)

    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id=video_id, languages=['en'])

    transcript = " ".join(chunk.text for chunk in transcript_list)

    print("\nTRANSCRIPT:\n")
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video")
except Exception as e:
    print("Error:", e)