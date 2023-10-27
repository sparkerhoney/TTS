import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch

class Model:
    def __init__(self):
        self.processor = None
        self.model = None
        self.vocoder = None
        self.speaker_embeddings = None

    @classmethod
    def from_path(cls, model_dir):
        model = cls()
        tts_model_dir = f"{model_dir}/tts_model"
        vocoder_model_dir = f"{model_dir}/vocoder_model"
        model.processor = SpeechT5Processor.from_pretrained(tts_model_dir)
        model.model = SpeechT5ForTextToSpeech.from_pretrained(tts_model_dir)
        model.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_model_dir)
        
        # Load speaker embeddings
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        model.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        
        return model

    def predict(self, text):
        inputs = self.processor(text=text, return_tensors="pt")
        with torch.no_grad():
            speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
        return speech.numpy(), 16000  
