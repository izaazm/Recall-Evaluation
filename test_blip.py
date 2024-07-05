import os
import time
import random
import requests
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, BlipTextModel, BlipVisionModel, AutoConfig, AutoModel, BlipForImageTextRetrieval

def set_seed(seed: int = 2) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def get_images(image_names):
    res = []
    for image_name in image_names:
        image_path = f"./CLIP4Cir/fashionIQ_dataset/images/{image_name}.png"
        res.append(Image.open(image_path))
    return res

def get_embeds(model, processor, src_text, src_image, tgt_images):
    # processing
    src_text_inputs = processor(text=src_text, return_tensors="pt")
    if src_image is not None:
        src_image_inputs = processor(images=src_image, return_tensors="pt")
    tgt_image_inputs = processor(images=tgt_images, return_tensors="pt")

    # image embedding
    tgt_image_outputs = model.vision_model(
        pixel_values=tgt_image_inputs.pixel_values,
        interpolate_pos_encoding=False
    )
    tgt_image_embeds = tgt_image_outputs[0]

    # multimodal embedding
    if src_image is not None:
        src_image_outputs = model.vision_model(
            pixel_values=src_image_inputs.pixel_values,
            interpolate_pos_encoding=False
        )
        src_image_embeds = src_image_outputs[0]
        src_image_atts = torch.ones(src_image_embeds.size()[:-1], dtype=torch.long)

        multimodal_embeds = model.text_encoder(
            input_ids=src_text_inputs.input_ids,
            attention_mask=src_text_inputs.attention_mask,
            encoder_hidden_states=src_image_embeds,
            encoder_attention_mask=src_image_atts,
        )
        multimodal_embeds = multimodal_embeds[0]
    else:
        multimodal_embeds = model.text_encoder(
            input_ids=src_text_inputs.input_ids,
            attention_mask=src_text_inputs.attention_mask,
        )
        multimodal_embeds = multimodal_embeds[0]

    image_feat = F.normalize(model.vision_proj(tgt_image_embeds[:, 0, :]), dim=-1)
    multimodal_feat = F.normalize(model.text_proj(multimodal_embeds[:, 0, :]), dim=-1)

    output = multimodal_feat @ image_feat.t()
    return output.detach().cpu().numpy().squeeze().tolist()

if __name__ == "__main__":
    print("Hello, World!")

    if torch.cuda.is_available():
        print("Cuda is available")
        device = torch.device("cuda")
    else:
        print("Cuda is not available")
        device = torch.device("cpu")

    processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
    model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco")

    # Batched result
    time_start = time.time()
    source_image = Image.open("./CLIP4Cir/fashionIQ_dataset/images/B0083I6W08.png")
    retrieved_images_names = ['B00BPD4N5E', 'B00BIQKAWS', 'B001THROSE', 'B008R567RU', 'B00A3F9MS8', 'B00BHKFFAW', 'B00CLCHVSY', 'B006L28DQY', 'B000LZO27G', 'B0077PMHIO', '9830019934']
    retrieved_images = get_images(retrieved_images_names)
    source_text = "Is green with a four leaf clover and is green and has no text"

    scores = get_embeds(model, processor, source_text, source_image, retrieved_images)
    print("Batched Result: ")
    print(["{0:0.2f}".format(score) for score in scores], time.time() - time_start)

    # Single result
    time_start = time.time()
    source_image = Image.open("./CLIP4Cir/fashionIQ_dataset/images/B0083I6W08.png")
    source_text = "Is green with a four leaf clover and is green and has no text"
    scores = []
    for retrieved_image in retrieved_images_names:
        tgt_image_path = f"./CLIP4Cir/fashionIQ_dataset/images/{retrieved_image}.png"
        tgt_image = Image.open(tgt_image_path)
        score = get_embeds(model, processor, source_text, source_image, tgt_image)
        scores.append(score)
    print("Single Result: ")
    print(["{0:0.2f}".format(score) for score in scores], time.time() - time_start)