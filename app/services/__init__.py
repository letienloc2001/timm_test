from .classification import Classification
from .detection import InferenceDetection
from .quality_image import ImageQuality

__all__ = ["InferenceClassification", "InferenceDetection", "ImageQuality"]


def mapping(key, value):
    try:
        parse_syntax = "{}('{}')".format(value['instance'], value['checkpoint_path'])
        return eval(parse_syntax)
    except:
        raise IndexError("Please check your configuration:", key)
    
def init_models(models_config):
    return {key: mapping(key, value) for key, value in models_config.items()}
