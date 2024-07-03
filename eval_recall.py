import os
import gc
import sys
from operator import itemgetter

import numpy as np
from PIL import Image
from tqdm import tqdm

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoTokenizer, BlipModel

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

@torch.no_grad()
def get_retrieved_image_score(blip_processor, blip_model, device, src_text, src_image, retrieved_images):
    src_inputs = blip_processor(images=get_images([src_image]), text=src_text, padding=True, return_tensors="pt").to(device)
    src_features = blip_model.get_multimodal_features(**src_inputs).detach().cpu()

    tgt_inputs = blip_processor(images=get_images(retrieved_images), return_tensors="pt").to(device)
    tgt_features = blip_model.get_image_features(**tgt_inputs).detach().cpu()

    score = (torch.sum(F.cosine_similarity(src_features, tgt_features)) / len(retrieved_images)) * 100
    return score


def get_clip4cir_model(clip_path, model_name, combiner_path, device):
    print("Getting Clip Model")

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


def fashioniq_retrieval(dress_type, clip_model, combining_function, preprocess, blip_backbone, device):
    classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
    relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_model)

    return get_metric(relative_val_dataset, clip_model, index_features, index_names, combining_function, blip_backbone, device)


def get_metric(relative_val_dataset, clip_model, index_features, index_names, combining_function, blip_backbone, device):
    predicted_features, target_names, captions = get_preds(relative_val_dataset, clip_model, index_features, index_names, combining_function)
    print(f"Predicted features shape : {predicted_features.shape}") # val_entry x feature_shape
    print(f"Target names shape : {len(target_names)}") # val_entry x feature_shape
    len_data = len(target_names)

    # Free up memory
    del clip_model, combining_function
    _ = gc.collect()

    print(f"Computing FashionIQ {relative_val_dataset.dress_types} validation metrics")

    print("Getting BLIP Model")
    blip_processor = AutoProcessor.from_pretrained(blip_backbone)
    blip_model = BlipModel.from_pretrained("./blip").to(device)
    print("Finshed Loading BLIP Model")

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    print(f"Predicted features shape : {predicted_features.shape}") # val_entry x feature_shape
    print(f"Index features shape : {index_features.shape}") # all_entry x feature_shape

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T # val_entry x all_entry
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    scoreat10 = 0
    scoreat50 = 0
    for i in tqdm(range(sorted_index_names.shape[0])):
        scoreat10 += get_retrieved_image_score(blip_processor, blip_model, device, captions[i], target_names[i], sorted_index_names[i][:10])
        scoreat50 += get_retrieved_image_score(blip_processor, blip_model, device, captions[i], target_names[i], sorted_index_names[i][:50])

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    blip_scores_at10 = scoreat10 / len(labels)
    blip_scores_at50 = scoreat50 / len(labels)

    return recall_at10, recall_at50, blip_scores_at10, blip_scores_at50

@torch.no_grad()
def get_preds(relative_val_dataset, clip_model, index_features, index_names, combining_function):
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                    num_workers=1, pin_memory=True, shuffle=False)
    name_to_feat = dict(zip(index_names, index_features))

    predicted_features = torch.empty((0, clip_model.visual.output_dim)).to(device, non_blocking=True)
    target_names = []
    captions_test = []

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
        target_names.extend(batch_target_names)
        captions_test.extend(input_captions)

    return predicted_features, target_names, captions_test


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'RN50'
    clip_path = "./CLIP4Cir/pretrained/fiq_clip_RN50_fullft.pt"
    combiner_path = "./CLIP4Cir/pretrained/fiq_comb_RN50_fullft.pt"
    blip_backbone = "Salesforce/blip-itm-base-coco"
    clip_model, preprocess, combining_function = get_clip4cir_model(clip_path, model_name, combiner_path, device)

    ## Test on FashionIQ dataset's
    recallat10, recallat50 = fashioniq_retrieval('shirt', clip_model, combining_function, preprocess, blip_backbone, device)
    print(f"Recall @ 10 : {recallat10}")
    print(f"Recall @ 50 : {recallat50}")