import whisper
import os
import requests
from pydub import AudioSegment

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL =  os.getenv("WHISPER_MODEL","small") # apn ne default value small di hai agar user ne .env file me koi value set nahi ki to. Whisper model ke different sizes hote hain jaise tiny, base, small, medium, large. Small model ek achha balance provide karta hai speed aur accuracy ke beech me, isliye maine isse default set kiya hai.

# we have to create end points to use sarvam
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY") 
SARVAM_STT_TRANSLATE_URL= "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL","saaras:v2.5")


_model = None  # jab ham pehli baar function use karenge to vo hamare system me download nahi rahega, to ham is variable ko None se initialize karenge. Jab ham load_model function call karenge, to ham check karenge ki model variable me koi value hai ya nahi. Agar model variable None hai, to ham whisper.load_model function ka use karke specified model ko load karenge aur usse model variable me store karenge. Is tarah se ham ensure karenge ki model sirf ek baar load ho aur subsequent calls me ham already loaded model ka use karenge, jisse performance improve hogi.


# we want to load model once only
def load_model():
    global _model
    if _model is None:
        print(f"loading Model ...")
        _model = whisper.load_model(WHISPER_MODEL)
    return _model

# now model is used to transcribe the chunks
def transcribe_chunk_whisper(chunk_path:str , translate:bool = False) -> str: # translate parameter ka use karne ka reason ye hai ki whisper model me ek feature hota hai jise "translate" kehte hain. Jab aap is parameter ko True set karte hain, to model automatically detected language ko English me translate kar deta hai. Iska fayda ye hai ki agar aapke audio me koi aur language hai jo English nahi hai, to bhi aapko English me transcription result milega. Agar aapko original language me transcription chahiye, to aap is parameter ko False rakh sakte hain. By default, maine is parameter ko False set kiya hai, lekin aap apne requirement ke hisab se ise change kar sakte hain.
    model = load_model()
    result = model.transcribe(chunk_path, task="transcribe")
    return result['text']

def _send_to_sarvam(piece_path: str) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n Sarvam returned {response.status_code}")
        print(f"Response body: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "")

def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""
    total_pieces =  (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


# for videos we have more than one chunks to transcribe, so we will use this function to transcribe all the chunks and combine the results into one string. Is function me ham ek list of chunk paths pass karenge, aur ye function har chunk ko transcribe karega aur unke results ko ek single string me combine karke return karega. Is tarah se ham apne video ke poore audio ka transcription result ek hi string me le sakte hain, chahe usme kitne bhi chunks ho.
def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  → Whisper (local model)
    - hinglish → Sarvam (translates to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)
def transcribe_all(chunks: list, language: str = "english") -> str:

    full_transcript = "" 

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):  

        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        text = transcribe_chunk(chunk, language=language)  

        full_transcript += text + " "  

    print("Transcription complete.")

    return full_transcript.strip()
