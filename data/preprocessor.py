import torch
from transformers import LayoutLMv3Processor, DonutProcessor
from PIL import Image

class LayoutLMPreprocessor:
    def __init__(self, model_checkpoint="microsoft/layoutlmv3-base"):
        self.processor = LayoutLMv3Processor.from_pretrained(model_checkpoint, apply_ocr=False)
    
    def __call__(self, examples):
        images = [image.convert("RGB") for image in examples["image"]]
        words = examples["words"]
        boxes = examples["bboxes"]
        word_labels = examples["ner_tags"]

        encoding = self.processor(
            images, 
            words, 
            boxes=boxes, 
            word_labels=word_labels,
            truncation=True, 
            padding="max_length",
            max_length=512
        )
        return encoding

class DonutPreprocessor:
    def __init__(self, model_checkpoint="naver-clova-ix/donut-base"):
        self.processor = DonutProcessor.from_pretrained(model_checkpoint)
    
    def __call__(self, image: Image.Image, prompt=None):
        """
        Prepares image and prompt for Donut.
        """
        if prompt:
            pixel_values = self.processor(image, prompt=prompt, return_tensors="pt").pixel_values
        else:
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
        return pixel_values

def normalize_bbox(bbox, width, height):
    """
    Normalizes bounding boxes to 0-1000 range.
    FUNSD bboxes are already in pixels, but LayoutLM requires normalization.
    """
    return [
        int(1000 * (bbox[0] / width)),
        int(1000 * (bbox[1] / height)),
        int(1000 * (bbox[2] / width)),
        int(1000 * (bbox[3] / height)),
    ]
