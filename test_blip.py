import os
import random

import requests
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, BlipTextModel, BlipVisionModel, AutoConfig

def set_seed(seed: int = 2) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == "__main__":
    print("Hello, World!")    

    if torch.cuda.is_available():
        print("Cuda is available")
        device = torch.device("cuda")
    else:
        print("Cuda is not available")
        device = torch.device("cpu")

    blipconfig = AutoConfig.from_pretrained("Salesforce/blip-itm-base-coco")
    blipvisionmodel = BlipVisionModel.from_config(blipconfig.vision_config)
    bliptextmodel = BlipTextModel.from_config(blipconfig.text_config, add_pooling_layer=False)
    image_processor = AutoProcessor.from_config(blipconfig)

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs_image = image_processor(images=image, return_tensors="pt").to(device)
    inputs_text = text_processor(text=["a photo of a cat", "a photo of two cats"], return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        image_features = F.normalize(model.get_image_features(**inputs_image).detach().cpu(), -1)
        print(image_features.shape)
        text_features = F.normalize(model.get_text_features(**inputs_text).detach().cpu(), -1)
        print(text_features.shape)
    
    logit = F.cosine_similarity(image_features, text_features)
    print(logit)
