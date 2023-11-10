from flask import Flask, request, jsonify
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from transformers import pipeline
import soundfile as sf
from datasets import load_dataset

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")

app = Flask(__name__)

# Initialize SpeechT5 synthesizer
synthesizer = pipeline("text-to-speech", model="microsoft/speecht5_tts")

@app.route('/tts', methods=['POST'])
def tts():
    data = request.json
    input_text = data.get('text', '')
    
    # Load speaker embedding
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    
    # Generate speech
    speech = synthesizer(input_text, forward_params={"speaker_embeddings": speaker_embedding})
    
    # Save the speech to a file
    sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])
    
    # Respond with success message and file path
    response = {
        'message': 'Speech generated successfully',
        'file': 'speech.wav'
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)