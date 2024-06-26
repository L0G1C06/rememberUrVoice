from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
import shutil
import os 

from utils import ttsText2Audio, executeAudio

router = APIRouter(prefix="/audio", tags=["audio"])

@router.post("/generate")
async def text2Audio(text: str = Form(...), speaker_wav: UploadFile = File(...)):
    output_dir = "./output/"
    os.makedirs(output_dir, exist_ok=True)
    speaker_wav_path = f"temp_{speaker_wav.filename}"
    output_file_path = os.path.join(output_dir, "output.wav")

    with open(speaker_wav_path, "wb") as buffer:
        shutil.copyfileobj(speaker_wav.file, buffer)

    ttsText2Audio(text, speaker_wav_path, output_file_path)

    os.remove(speaker_wav_path)

    return FileResponse(output_file_path, media_type="audio/wav", filename="output.wav")

@router.post("/reproduce")
async def reproduceAudio(wav_file: UploadFile = File(...)):
    speaker_wav_path = f"temp_{wav_file.filename}"

    with open(speaker_wav_path, "wb") as buffer:
        shutil.copyfileobj(wav_file.file, buffer)

    executeAudio(speaker_wav_path)

    os.remove(speaker_wav_path)
    return JSONResponse(content="", status_code=200)