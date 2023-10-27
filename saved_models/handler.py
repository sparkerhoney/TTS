from abc import ABC
import logging
import torch
import torchaudio
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from ts.torch_handler.base_handler import BaseHandler
from datasets import load_dataset


logger = logging.getLogger(__name__)

class SpeechSynthesisHandler(BaseHandler, ABC):
    def __init__(self):
        super(SpeechSynthesisHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu"
        )
        # Load the models from the saved directory
        self.processor = SpeechT5Processor.from_pretrained(model_dir)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(model_dir)
        self.vocoder = SpeechT5HifiGan.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        logger.debug("Speech synthesis model from path {0} loaded successfully".format(model_dir))
        self.initialized = True

    def preprocess(self, data):
        text = [d["data"] for d in data]
        logger.info("Received text example: '%s'", text[0])
        inputs = self.processor(text=text, padding=True, truncation=True, return_tensors="pt")
        return inputs
    
    def inference(self, inputs):
        inputs = inputs.to(self.device)
        
        # Assuming you have the embeddings_dataset loaded globally or load it here
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation", cache_dir="/path_to_cache_dir")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        
        with torch.no_grad():
            speech = self.model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=self.vocoder)
        return speech

    
    def postprocess(self, inference_output):
        # Define the file path for the output WAV file
        file_path = "output.wav"
        
        # Add a new dimension to make the tensor 2D
        # Assuming inference_output is 1D, with shape (num_samples,)
        # The new shape will be (1, num_samples)
        inference_output_2d = inference_output.unsqueeze(0)
        
        # Save the tensor to a file
        torchaudio.save(file_path, inference_output_2d, sample_rate=22050)
        
        # Return the file path for further use
        return file_path
    
_service = SpeechSynthesisHandler()

def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)
        if data is None:
            return None
        data = _service.preprocess(data)
        data = _service.inference(data)
        data = _service.postprocess(data)
        return data
    except Exception as e:
        raise e
