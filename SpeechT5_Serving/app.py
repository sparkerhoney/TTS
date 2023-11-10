from flask import Flask, request, jsonify
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf
from datasets import load_dataset

app = Flask(__name__)

# Initialize SpeechT5 synthesizer
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

@app.route('/tts', methods=['POST'])
def tts():
    data = request.json
    input_text = data.get('text', '')
    
    # Load speaker embedding
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    
    # Preprocess input text
    input_ids = processor(text=input_text, return_tensors="pt").input_ids
    
    # Generate speech
    speech = model.generate_speech(input_ids, speaker_embedding, vocoder=vocoder)
    
    # Save the speech to a file
    sf.write("speech.wav", speech.numpy(), samplerate=16000)
    
    # Respond with success message and file path
    response = {
        'message': 'Speech generated successfully',
        'file': 'speech.wav'
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)