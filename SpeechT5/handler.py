# import soundfile as sf
# from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
# from datasets import load_dataset
# import torch

# class Model:
#     def __init__(self):
#         self.processor = None
#         self.model = None
#         self.vocoder = None
#         self.speaker_embeddings = None

#     @classmethod
#     def from_path(cls, model_dir):
#         model = cls()
#         tts_model_dir = f"{model_dir}/tts_model"
#         vocoder_model_dir = f"{model_dir}/vocoder_model"
#         model.processor = SpeechT5Processor.from_pretrained(tts_model_dir)
#         model.model = SpeechT5ForTextToSpeech.from_pretrained(tts_model_dir)
#         model.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_model_dir)
        
#         # Load speaker embeddings
#         embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
#         model.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        
#         return model

#     def predict(self, text):
#         inputs = self.processor(text=text, return_tensors="pt")
#         with torch.no_grad():
#             speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
#         return speech.numpy(), 16000  

import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
from ts.torch_handler.base_handler import BaseHandler

class CustomTTSHandler(BaseHandler):
    def __init__(self):
        self._context = None
        self.initialized = False

    def initialize(self, context):
        self._context = context
        self.initialized = True
        
        self.model_dir = context.system_properties.get("model_dir")
        tts_model_dir = f"{self.model_dir}/tts_model"
        vocoder_model_dir = f"{self.model_dir}/vocoder_model"
        self.processor = SpeechT5Processor.from_pretrained(tts_model_dir)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(tts_model_dir)
        self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_model_dir)

        # Load speaker embeddings
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    def preprocess(self, data):
        text = data[0].get("data")
        if text is None:
            text = data[0].get("body")
        inputs = self.processor(text=text, return_tensors="pt")
        return inputs

    def inference(self, model_input):
        with torch.no_grad():
            speech = self.model.generate_speech(model_input["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
        return speech.numpy(), 16000  # numpy array와 sample rate를 반환합니다.

    def postprocess(self, inference_output):
        speech_output, samplerate = inference_output
        return {"speech": speech_output.tolist(), "samplerate": samplerate}

    def handle(self, data, context):
        model_input = self.preprocess(data)
        inference_output = self.inference(model_input)
        return self.postprocess(inference_output)
