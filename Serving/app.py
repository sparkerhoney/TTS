from flask import Flask, send_file
from TTS.utils.synthesize import Synthesizer

app = Flask(__name__)
synthesizer = Synthesizer("tts_models/en/ljspeech/glow-tts", "vocoder_models/en/ljspeech/multiband-melgan")

@app.route('/synthesize')
def synthesize():
    text = "hello world"
    wav = synthesizer.tts(text)
    synthesizer.save_wav(wav, "output.wav")
    return send_file("output.wav")

if __name__ == '__main__':
    app.run()