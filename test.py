from utils.audio_processor import process_input
from core.transcriber import transcribe_all

source = "https://www.youtube.com/watch?v=tplWXd_T7YQ" # example youtube video url
chunks = process_input(source)
print(transcribe_all(chunks))
# hindi is also transcripted but not as good as english by whisper model, so we can use translate=True to get better results in english.