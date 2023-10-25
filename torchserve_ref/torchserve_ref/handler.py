from abc import ABC
import json
import logging
import os
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class CorrectionGenerationHandler(BaseHandler, ABC):
    def __init__(self):
        super(CorrectionGenerationHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu"
        )
        # Read model serialize/pt file
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        logger.debug("Correction generation model from path {0} loaded successfully".format(model_dir))

        # Read the mapping file, index to object name
        generation_config_file_path = os.path.join(model_dir, "generation_config.json")
        if os.path.isfile(generation_config_file_path):
            with open(generation_config_file_path) as f:
                self.generation_config = json.load(f)
        else:
            logger.warning(
                "Missing the generation_config.json file."
            )
        self.initialized = True

    def preprocess(self, data):
        """
        Tokenizes input text.
        Args:
            data (list): a list of dictionary objects that contain student's last utterances without name tags ("Student:", "User:", etc.)
        Returns:
            inputs (dict): tokenized text. Keys are: 'input_ids' and 'attention_mask'.
        """
        # TODO: Whether or not you are using a batch, make sure that 'text' is a list of string(s).
        # TODO: if any utterance string has a name tag("Student:", "User:", etc.), remove it.
        try:
            text = [d["data"] for d in data]
        except:
            text = [d["body"] for d in data]
        logger.info("Received text example: '%s'", text[0])
        
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        return inputs
    
    def inference(self, inputs):
        """
        Generates error correction result.
        Args:
            inputs (dict): contains tokenized text. Keys are: 'input_ids' and 'attention_mask'
        Returns:
            correction (list): a list that contains the error correction(str).
        """
        inputs = inputs.to(self.device)

        # You may customize the generation config file
        if self.generation_config:
            output = self.model.generate(**inputs, **self.generation_config)
        else:
            output = self.model.generate(**inputs)

        correction = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        logger.info("Model correction example: '%s'", correction[0])

        return correction
    
    def postprocess(self, inference_output):
        # TODO: Add any needed post-processing of the model predictions here
        # TODO: If you want are using an instance, not a batch, and want to get a sentence-level output, simply index [0] here.
        logger.info("Model Name: '%s'", self.model.config._name_or_path)
        return inference_output
    
_service = CorrectionGenerationHandler()

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