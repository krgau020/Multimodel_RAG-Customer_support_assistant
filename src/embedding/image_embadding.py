# Image embedding using CLIP

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(image_path, debug=True):
    """
    Generate image embedding using CLIP.
    """
    img = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=img, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    embedding = features.squeeze().numpy()

    if debug:
        print(f"\n[DEBUG] Image path: {image_path}")
        print(f"Image Embedding shape: {embedding.shape}")
        print(f"Embedding sample: {embedding[:5]}")

    return embedding
