from PIL import Image
import requests
import torch
from transformers import AutoProcessor, CLIPModel

if __name__ == "__main__":
    print("Hello, World!")    
    if torch.cuda.is_available():
        print("Cuda is available")
        device = torch.device("cuda")
    else:
        print("Cuda is not available")
        device = torch.device("cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = processor(
        text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
    ).to(device)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).detach().cpu().numpy()
    print(probs)
