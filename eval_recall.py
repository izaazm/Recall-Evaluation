import os
import gc
import sys
from operator import itemgetter
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer, BlipForImageTextRetrieval

from CLIP4Cir.src.data_utils import targetpad_transform, FashionIQDataset
from CLIP4Cir.src.combiner import Combiner
from CLIP4Cir.src.utils import extract_index_features


CONFIG = {
    "target_ratio": 1.25,
    "projector_dim": 4096, 
    "hidden_dim": 8192
}


def get_images(image_names):
    res = []
    for image_name in image_names:
        image_path = f"CLIP4Cir/fashionIQ_dataset/images/{image_name}.png"
        res.append(Image.open(image_path))
    return res

def get_clip4cir_model(clip_path, model_name, combiner_path, device):
    print("Getting Clip4cir Model")

    clip_model, _ = clip.load(model_name, jit=False)
    input_dim = clip_model.visual.input_resolution
    preprocess = targetpad_transform(CONFIG["target_ratio"], input_dim)
    clip_state_dict = torch.load(clip_path)
    clip_model.load_state_dict(clip_state_dict["CLIP"])
    clip_model = clip_model.to(device).float().eval()

    feature_dim = clip_model.visual.output_dim
    combiner = Combiner(feature_dim, CONFIG["projector_dim"], CONFIG["hidden_dim"])
    combiner_state_dict = torch.load(combiner_path)
    combiner.load_state_dict(combiner_state_dict['Combiner'])
    combiner.to(device).eval()
    combining_function = combiner.combine_features
    
    print("Finished Loading Model")
    return clip_model, preprocess, combining_function

def get_blip_model(backbone, device):
    print("Getting BLIP model")
    blip_processor = AutoProcessor.from_pretrained(backbone)
    blip_model = BlipForImageTextRetrieval.from_pretrained(backbone).to(device).eval()
    print("Finished Loading model")
    return blip_processor, blip_model

@torch.no_grad()
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


def get_correlation_retrieval(index_features, index_names, predicted_features, target_names, reference_names, captions, blip_processor, blip_model, device):
    len_data = len(target_names)

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T # val_entry x all_entry
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    spearman_corrs, spearman_pvalues = [], []
    pearson_corrs, pearson_pvalues = [], []

    for i in tqdm(range(sorted_index_names.shape[0])):

        scores =  get_blip_scores(blip_model, blip_processor, captions[i], get_images([reference_names[i]]), get_images(sorted_index_names[i][:100]), device)
        spearman_corr, spearman_pvalue = spearmanr(scores, range(100))
        pearson_corr, pearson_pvalue = pearsonr(scores, range(100))
        spearman_corrs.append(spearman_corr)
        spearman_pvalues.append(spearman_pvalue)
        pearson_corrs.append(pearson_corr)
        pearson_pvalues.append(pearson_pvalue)

    spearman_corr = np.mean(spearman_corrs)
    spearman_pvalue = np.mean(spearman_pvalues)
    pearson_corr = np.mean(pearson_corrs)
    pearson_pvalue = np.mean(pearson_pvalues)

    return spearman_corr, spearman_pvalue, pearson_corr, pearson_pvalue

@torch.no_grad()
def get_preds(relative_val_dataset, clip_model, index_features, index_names, combining_function):
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                    num_workers=1, pin_memory=True, shuffle=False)
    name_to_feat = dict(zip(index_names, index_features))

    predicted_features = torch.empty((0, clip_model.visual.output_dim)).to(device, non_blocking=True)
    target_names = []
    captions_test = []
    all_reference_names = []

    for reference_names, batch_target_names, captions in tqdm(relative_val_loader):

        # Concatenate the captions in a deterministic way
        flattened_captions: list = np.array(captions).T.flatten().tolist()
        input_captions = [
            f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
            i in range(0, len(flattened_captions), 2)]
        text_inputs = clip.tokenize(input_captions, context_length=77).to(device, non_blocking=True)
        # Compute the predicted features
        with torch.no_grad():
            text_features = clip_model.encode_text(text_inputs)
            # Check whether a single element is in the batch due to the exception raised by torch.stack when used with
            # a single tensor
            if text_features.shape[0] == 1:
                reference_image_features = itemgetter(*reference_names)(name_to_feat).unsqueeze(0)
            else:
                reference_image_features = torch.stack(itemgetter(*reference_names)(
                    name_to_feat))  # To avoid unnecessary computation retrieve the reference image features directly from the index features
            batch_predicted_features = combining_function(reference_image_features, text_features)

        predicted_features = torch.vstack((predicted_features, F.normalize(batch_predicted_features, dim=-1)))
        all_reference_names.extend(reference_names)
        target_names.extend(batch_target_names)
        captions_test.extend(input_captions)

    return predicted_features, target_names, all_reference_names, captions_test


def fashioniq_retrieval(dress_type, clip_model, combining_function, preprocess):
    classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
    relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_model)

    predicted_features, target_names, reference_names, captions = get_preds(relative_val_dataset, clip_model, index_features, index_names, combining_function)
    return index_features, index_names, predicted_features, target_names, reference_names, captions

def parse_arguments():
    parser = argparse.ArgumentParser(description='FashionIQ Retrieval Script')
    parser.add_argument('--dress_type', type=str, required=True, help='Type of dress for retrieval')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    dress_type = args.dress_type

    model_name = 'RN50'
    clip_path = "./CLIP4Cir/pretrained/fiq_clip_RN50_fullft.pt"
    combiner_path = "./CLIP4Cir/pretrained/fiq_comb_RN50_fullft.pt"
    blip_backbone = "Salesforce/blip-itm-large-coco"
    clip_model, preprocess, combining_function = get_clip4cir_model(clip_path, model_name, combiner_path, device)

    # Test on FashionIQ dataset's
    index_features, index_names, predicted_features, target_names, reference_names, captions = fashioniq_retrieval(
        dress_type, clip_model, combining_function, preprocess
    )

    ## Free up Memory
    del clip_model, combining_function
    _ = gc.collect()

    blip_processor, blip_model = get_blip_model(blip_backbone, device)
    spearman_corr, spearman_pvalue, pearson_corr, pearson_pvalue = get_correlation_retrieval(
        index_features, index_names, predicted_features, target_names, reference_names, captions, blip_processor, blip_model, device
    )
    print(f"Spearman Correlation: {spearman_corr}, P-Value: {spearman_pvalue}")
    print(f"Pearson Correlation: {pearson_corr}, P-Value: {pearson_pvalue}")