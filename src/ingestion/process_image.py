#process images to generate captions for multimodal retrieval

from PIL import Image
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load BLIP model once
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def preprocess_image(image_path, debug=True):
    """
    Generate a caption for an image using BLIP (for LLM context).
    """
    img = Image.open(image_path).convert("RGB")

    # Generate caption
    inputs = processor(img, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    if debug:
        print(f"\n[DEBUG] Processed image from: {image_path}")
        print(f"[DEBUG] Image caption: {caption}")

    return caption
