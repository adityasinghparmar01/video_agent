import yt_dlp # for downloading audio from YouTube
from pydub import AudioSegment # for processing audio files & chunking
import os

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# from youtube vali videos ko download karne ke liye hum yt_dlp library ka use karenge jo ki ek powerful tool hai YouTube se audio aur video download karne ke liye. Hum is function me YouTube video ka URL pass karenge aur ye function us video se audio ko extract karke WAV format me save karega.

def download_youtube_audio(url :str) ->str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")
    ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": output_path,
    "quiet": True,
    "noplaylist": True,
    "nocheckcertificate": True,
    "geo_bypass": True,
    "extractor_args": {
        "youtube": {
            "player_client": ["android", "web"]
        }
    },
    "http_headers": {
        "User-Agent": "Mozilla/5.0"
    },
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "wav",
        "preferredquality": "192",
    }],
}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".wav").replace(".m4a", ".wav")
    return filename

# ham covert karenge mp3/mp4/youtube etc audio files to wav format & good monostable 16khz audio for processing and chunking for using in whisper model for transcription. Hum is function me kisi bhi audio/video file ko WAV format me convert karenge jise hum apne transcription process me use kar sakte hain. Is function me hum pydub library ka use karenge jo ki audio processing ke liye bahut hi useful hai. Hum audio ko mono channel aur 16kHz sample rate me convert karenge jo ki whisper model ke liye ideal hai.

def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000) #16khz
    audio.export(output_path, format="wav")
    return output_path

# audio -> into chunks

def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path) # file load ho  gai
    chunk_ms = chunk_minutes*60*1000 # convert minutes to milliseconds -> b/c it works in ms

    chunks = []

    for i , start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start+chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)
    return chunks

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks

# we can use model of whisper ai locally for transcription and for that we need to process the audio in the right format and chunk it into smaller pieces so that it can be fed into the model for transcription. The above code provides functions to download audio from YouTube, convert any audio/video file to WAV format, and chunk the audio into smaller pieces for processing. The `process_input` function takes care of determining whether the input is a YouTube URL or a local file and processes it accordingly.