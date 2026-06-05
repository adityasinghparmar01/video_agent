import os
import yt_dlp # downloading audio from youtube links 
from pydub import AudioSegment # audio processing and chunking

def download_youtube_audio(url):
    os.makedirs("downloads", exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "downloads/%(id)s.%(ext)s",
        "restrictfilenames": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        file_path = ydl.prepare_filename(info)

        if not os.path.exists(file_path):
          downloaded_files = os.listdir("downloads")
        if downloaded_files:
               file_path = os.path.join("downloads", downloaded_files[0])

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
    audio = AudioSegment.from_file(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start:start + chunk_ms]
        chunk_path = f"{os.path.splitext(wav_path)[0]}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks

def process_input(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        downloaded_path = download_youtube_audio(source)

        print("Converting YouTube audio to WAV...")
        wav_path = convert_to_wav(downloaded_path)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)

    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks

# we can use model of whisper ai locally for transcription and for that we need to process the audio in the right format and chunk it into smaller pieces so that it can be fed into the model for transcription. The above code provides functions to download audio from YouTube, convert any audio/video file to WAV format, and chunk the audio into smaller pieces for processing. The `process_input` function takes care of determining whether the input is a YouTube URL or a local file and processes it accordingly.