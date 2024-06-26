from TTS.api import TTS
import playsound
import os

def ttsText2Audio(text, speaker_wav, output_path):
    accept_tos = os.getenv('ACCEPT_TOS', '').lower() in ['true', '1']
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    tts.tts_to_file(text, speaker_wav=speaker_wav, language="pt", file_path=output_path)

def executeAudio(wav_file):
    playsound.playsound(wav_file, block=True)