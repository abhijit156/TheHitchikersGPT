import whisper
import os
import time

import torch
from transformers import pipeline
from faster_whisper import WhisperModel


## Commenting out first approach which does not use the faster whisper model
# model = whisper.load_model('medium')
# def transcribe_audio(file_path):
#     try:
#         with open(file_path, 'rb') as file:
#             audio_data = file.read()
#             result = model.transcribe(file_path)
#             print(result['text'])
#             return result['text']
#     except Exception as e:
#         return f"Error transcribing {file_path}: {str(e)}"

# List all audio files in the folder
model_size = "large-v2"
# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

folder_path = 'audiobooks/'
output_txt_file = 'transcriptions.txt'

audio_files = [f for f in os.listdir(folder_path) f.endswith(('.mp3', '.wav', '.flac'))] #Subset-if f.startswith("00_") and


# Using HuggingFace pipeline to make life easier
pipe = pipeline("automatic-speech-recognition",
                "openai/whisper-large-v2",
                # torch_dtype=torch.float16,
                device="cuda:0")
                
start = time.time()
with open(output_txt_file, 'wb') as txt_file:
    for audio_file in audio_files:
        audio_path = os.path.join(folder_path, audio_file)
        transcription = pipe(audio_path,batch_size=16)
        text = transcription['text']
        # txt_file.write(transcription['text'].encode('utf-8'))
        txt_file.write((text+u"\n").encode('utf-8'))

        print("Transcribed " + str(audio_file))
        pipe.call_count = 0

print("Time taken : " + str(time.time() - start))
