from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from urllib.parse import urlparse, parse_qs


def extract_video_id(url: str) -> str:
    """
    Extracts the YouTube video ID from any valid YouTube URL.
    Supports:
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


def get_youtube_transcript(url: str) -> str:
    """
    Given a YouTube URL, returns its English transcript as a single string.
    Raises helpful errors if the transcript is unavailable.
    """
    try:
        video_id = extract_video_id(url)

        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id=video_id,languages=['en'])

        # Join all text into one string
        transcript = " ".join(chunk["text"] for chunk in transcript_list)

        return transcript

    except TranscriptsDisabled:
        raise RuntimeError("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise RuntimeError("No English transcript found for this video.")
    except Exception as e:
        raise RuntimeError(f"Error retrieving transcript: {e}")