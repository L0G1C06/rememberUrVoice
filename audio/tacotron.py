import torch
import torchaudio 
import playsound

torch.random.manual_seed(42)
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"
print("Using: ", device)

class Tacotron:
    def __init__(self, wav_file, text):
        self.wav_file = wav_file
        self.text = text

    def execute_audio(self):
        playsound.playsound(self.wav_file, block=True)

    def tacotron(self):
        bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
        processor = bundle.get_text_processor()
        vocoder = bundle.get_vocoder().to(device)
        tacotron2 = bundle.get_tacotron2().to(device)
        with torch.inference_mode():
            processed, lengths = processor(self.text)
            processed, lengths = processed.to(device), lengths.to(device)
            output = tacotron2.infer(processed, lengths)
            spec, spec_lengths = output[0], output[1]
            waveforms, lengths = vocoder(spec, spec_lengths)
        torchaudio.save(self.wav_file, waveforms.cpu(), vocoder.sample_rate)
        execute_audio(self.wav_file)

def execute_audio(wav_file: str):
    playsound.playsound(wav_file, block=False)

def tacotron(text):
    bundle = torchaudio.pipelines.TACOTRON2_WAVERNN_PHONE_LJSPEECH
    processor = bundle.get_text_processor()
    vocoder = bundle.get_vocoder().to(device)
    tacotron2 = bundle.get_tacotron2().to(device)
    with torch.inference_mode():
        processed, lengths = processor(text)
        processed, lengths = processed.to(device), lengths.to(device)
        output = tacotron2.infer(processed, lengths)
        spec, spec_lengths = output[0], output[1]
        waveforms, lengths = vocoder(spec, spec_lengths)
    torchaudio.save("output_audio.wav", waveforms.cpu(), vocoder.sample_rate)
    execute_audio("output_audio.wav")