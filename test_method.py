import os
import time
import random
import requests
from PIL import Image

import lpips
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import AutoProcessor, AutoConfig, CLIPModel, BlipForImageTextRetrieval, ViTModel 

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
        image_path = f"../fashionIQ_dataset/images/{image_name}.png"
        res.append(Image.open(image_path).convert("RGB"))
    return res


def get_blip_scores(model, processor, src_text, src_image, tgt_images, device):
    # processing
    src_text_inputs = processor(text=src_text, return_tensors="pt").to(device)
    if src_image is not None:
        src_image_inputs = processor(images=src_image, return_tensors="pt").to(device)
    tgt_image_inputs = processor(images=tgt_images, return_tensors="pt").to(device)

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


def get_src_clip_embedding(model, processor, src_text, src_image, device):
    inputs = processor(images=src_image, text=src_text, return_tensors="pt").to(device)
    image_embeds = model.get_image_features(pixel_values=inputs.pixel_values)
    text_embeds = model.get_text_features(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    return image_embeds + text_embeds


def get_clip_scores(model, processor, src_text, src_image, tgt_images, device):
    tgt_inputs = processor(images=tgt_images, return_tensors="pt").to(device)
    tgt_embeds = model.get_image_features(pixel_values=tgt_inputs.pixel_values)
    src_embeds = get_src_clip_embedding(model, processor, src_text, src_image, device)

    src_embeds = src_embeds / src_embeds.norm(p=2, dim=-1, keepdim=True)
    tgt_embeds = tgt_embeds / tgt_embeds.norm(p=2, dim=-1, keepdim=True)

    scores = src_embeds @ tgt_embeds.t()
    return scores.detach().cpu().numpy().squeeze().tolist()

def get_target_scores(model, processor, target_image, retrieved_image, device):
    target_inputs = processor(images=target_image, return_tensors="pt").to(device)
    retrieved_image_inputs = processor(images=retrieved_image, return_tensors="pt").to(device)

    target_embeds = model(pixel_values=target_inputs.pixel_values).last_hidden_state[:, 0]
    retrieved_images_embeds = model(pixel_values=retrieved_image_inputs.pixel_values).last_hidden_state[:, 0]

    target_embeds = target_embeds / target_embeds.norm(p=2, dim=-1, keepdim=True)
    retrieved_images_embeds = retrieved_images_embeds / retrieved_images_embeds.norm(p=2, dim=-1, keepdim=True)

    scores = target_embeds @ retrieved_images_embeds.t()

    return scores.detach().cpu().numpy().squeeze().tolist()

def get_lpips_scores(model, transform, target_image, retrieved_image, device):
    loss = []
    for retrieved_img in retrieved_image:
        img1 = transform(target_image).unsqueeze(0).to(device)
        img2 = transform(retrieved_img).unsqueeze(0).to(device)
        loss.append(1 - model(img1, img2).item())
    return loss

if __name__ == "__main__":
    model_test = "lpips" # "clip" or "blip" or "vit" ir "lpips"
    print(f"Hello, World!, Testing {model_test}")
    if torch.cuda.is_available():
        print("Cuda is available")
        device = torch.device("cuda")
    else:
        print("Cuda is not available")
        device = torch.device("cpu")

    # Batched result
    time_start = time.time()
    source_image = Image.open("../fashionIQ_dataset/images/B0083I6W08.png").convert("RGB")
    target_image = Image.open("../fashionIQ_dataset/images/B00BPD4N5E.png").convert("RGB")
    retrieved_images_names = ['B00BPD4N5E', 'B00BIQKAWS', 'B001THROSE', 'B008R567RU', 'B00A3F9MS8', 'B00BHKFFAW', 'B00CLCHVSY', 'B006L28DQY', 'B000LZO27G', 'B0077PMHIO', '9822504462']
    retrieved_images = get_images(retrieved_images_names)
    source_text = "Is green with a four leaf clover and is green and has no text"
    
    if model_test == "blip":
        processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-large-coco")
        model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-large-coco").to(device).eval()
        scores = get_blip_scores(model, processor, source_text, source_image, retrieved_images, device)
        print("Batched Result: ")
        print(["{0:0.2f}".format(score) for score in scores], time.time() - time_start)

    elif model_test == "clip":
        processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        scores = get_clip_scores(model, processor, source_text, source_image, retrieved_images, device)
        print("Batched Result: ")
        print(["{0:0.2f}".format(score) for score in scores], time.time() - time_start)

    elif model_test == "vit":
        processor = AutoProcessor.from_pretrained("google/vit-base-patch16-224")
        model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device).eval()
        scores = get_target_scores(model, processor, target_image, retrieved_images, device)
        print("Batched Result: ")
        print(["{0:0.2f}".format(score) for score in scores], time.time() - time_start)

    elif model_test == "lpips":
        loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        transform = transforms.Compose(
            [transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        scores = get_lpips_scores(loss_fn_alex, transform, target_image, retrieved_images, device)
        print("Batched Result: ")
        print(["{0:0.2f}".format(score) for score in scores], time.time() - time_start)