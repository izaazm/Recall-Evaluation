import requests
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModel

if __name__ == "__main__":
    print("Hello, World!")    

    if torch.cuda.is_available():
        print("Cuda is available")
        device = torch.device("cuda")
    else:
        print("Cuda is not available")
        device = torch.device("cpu")

    model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(
        text=["a photo of a cat", "a photo of a dog", "a photo of a motorcycle"], images=image, return_tensors="pt", padding=True
    ).to(device)

    outputs = model(**inputs)
    logits = outputs.logits_per_image
    print(logits)
