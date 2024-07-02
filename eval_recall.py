import os
import gc
import sys
import numpy as np
from tqdm import tqdm
from operator import itemgetter

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


def get_image(img_name):


@torch.no_grad()
def get_retrieved_image_score(backbone, device, src_text, src_image, retrived_images):
    processor = AutoProcessor.from_pretrained(backbone)
    model = BlipModel.from_pretrained(backbone).to(device)

    src_inputs = processor(images=src_image, text=src_text, padding=True, return_tensors="pt").to(device)
    src_features = model.get_mutlimodal_features(**src_inputs)

    running_score = 0
    for retrived_image in retrived_images:
        retrived_inputs = processor(images=retrived_image, padding=True, return_tensors="pt").to(device)
        retrived_features = model.get_image_features(**retrived_inputs)
        score = nn.functional.cosine_similarity(src_features, retrived_features).item()
        running_score += score

    score = running_score / len(retrived_images)
    return score


def get_clip4cir_model(clip_path, combiner_path, device):
    print("Getting Clip Model")

    model_name = 'RN50'
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
    
    print("FInished Loading Model")
    return clip_model, preprocess, combining_function


def fashioniq_retrieval(dress_type, clip_model, combining_function, preprocess):
    classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
    relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)
    index_features, index_names = extract_index_features(classic_val_dataset, clip_model)

    return get_metric(relative_val_dataset, clip_model, index_features, index_names, combining_function)


def get_metric(relative_val_dataset, clip_model, index_features, index_names, combining_function):
    predicted_features, target_names = get_preds(relative_val_dataset, clip_model, index_features, index_names, combining_function)
    print(f"Compute FashionIQ {relative_val_dataset.dress_types} validation metrics")

    index_features = F.normalize(index_features, dim=-1).float()

    # Normalize the index features
    index_features = F.normalize(index_features, dim=-1).float()

    # Compute the distances and sort the results
    distances = 1 - predicted_features @ index_features.T
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Compute the ground-truth labels wrt the predictions
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100

    return recall_at10, recall_at50

@torch.no_grad()
def get_preds(relative_val_dataset, clip_model, index_features, index_names, combining_function):
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32,
                                    num_workers=1, pin_memory=True, shuffle=False)
    name_to_feat = dict(zip(index_names, index_features))

    predicted_features = torch.empty((0, clip_model.visual.output_dim)).to(device, non_blocking=True)
    target_names = []

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

    return predicted_features, target_names


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_path = "./CLIP4Cir/pretrained/fiq_clip_RN50_fullft.pt"
    combiner_path = "./CLIP4Cir/pretrained/fiq_comb_RN50_fullft.pt"
    blip_backbone = "Salesforce/blip-itm-base-coco"
    clip_model, preprocess, combining_function = get_clip4cir_model(clip_path, combiner_path, device)

    ## Test on FashionIQ dataset's
    recallat10, recallat50 = fashioniq_retrieval('shirt', clip_model, combining_function, preprocess)
    print(f"Recall @ 10 : {recallat10}")
    print(f"Recall @ 50 : {recallat50}")