import torch
from models.layoutlm_v3 import LayoutLMV3Model
from models.ocr_engine import get_ocr_engine
from PIL import Image

class LayoutLMPipeline:
    def __init__(self, model_path="microsoft/layoutlmv3-base", tesseract_path=None):
        self.engine = get_ocr_engine(tesseract_path)
        self.model_wrapper = LayoutLMV3Model(model_checkpoint=model_path)
        
    def process(self, image_path):
        """
        Runs OCR and LayoutLM v3 to extract entities.
        """
        # 1. Image loading
        image = Image.open(image_path).convert("RGB")
        
        # 2. OCR for tokens and boxes
        ocr_df = self.engine.extract_data(image_path)
        words = ocr_df.text.tolist()
        boxes = []
        for _, row in ocr_df.iterrows():
            # Normalized boxes: [x0, y0, x1, y1]
            boxes.append([row.left, row.top, row.left + row.width, row.top + row.height])
            
        # 3. Model Prediction
        # Note: In a production script, we'd ensure bbox normalization here
        encoding = self.model_wrapper.processor(
            image, 
            words, 
            boxes=boxes, 
            return_tensors="pt", 
            truncation=True, 
            padding="max_length"
        )
        
        logits = self.model_wrapper.forward(encoding)
        predictions = logits.argmax(-1).squeeze().tolist()
        
        # 4. Grouping (Simplified example)
        results = {}
        for i, pred in enumerate(predictions[:len(words)]):
            label = self.model_wrapper.model.config.id2label[pred]
            if label != "O":
                results[label] = results.get(label, "") + " " + words[i]
                
        return {k: v.strip() for k, v in results.items()}
