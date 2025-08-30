# text embedding using CLIP

import torch
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_text(text, debug=True):
    """
    Generate text embedding using CLIP.
    """
    inputs = clip_processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
    with torch.no_grad():
        features = clip_model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    embedding = features.squeeze().numpy()

    if debug:
        print(f"\n[DEBUG] Text: {text[:60]}...")
        print(f"Text Embedding shape: {embedding.shape}")
        print(f"Embedding sample: {embedding[:5]}")

    return embedding
