import torch
import torch.nn.functional as F
from transformers import AutoModel

model = AutoModel.from_pretrained("Salesforce/blip-itm-base-coco")
model.save_pretrained("blip", from_pt=True) 
