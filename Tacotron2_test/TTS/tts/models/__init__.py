from typing import Dict, List, Union

from TTS.utils.generic_utils import find_module


def setup_model(config: Union[dict, "Coqpit"], samples: Union[List[List], List[Dict]] = None) -> "BaseTTS":
    # Check if config is dictionary and access accordingly
    model_name = config['model'] if isinstance(config, dict) else config.model
    print(" > Using model: {}".format(model_name))
    
    # fetch the right model implementation.
    base_model_name = None
    if "base_model" in config:
        base_model_name = config['base_model'] if isinstance(config, dict) else config.base_model
    if base_model_name is not None:
        MyModel = find_module("TTS.tts.models", base_model_name.lower())
    else:
        MyModel = find_module("TTS.tts.models", model_name.lower())
    model = MyModel.init_from_config(config=config, samples=samples)
    return model


