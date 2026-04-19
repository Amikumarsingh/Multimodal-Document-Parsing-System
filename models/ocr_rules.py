import easyocr
import numpy as np
from PIL import Image
import re

class OCRRuleBasedParser:
    def __init__(self, languages=['en']):
        self.reader = easyocr.Reader(languages)
    
    def apply_ocr(self, image_path_or_pil):
        """
        Applies EasyOCR to the image.
        """
        if isinstance(image_path_or_pil, Image.Image):
            image_np = np.array(image_path_or_pil)
        else:
            image_np = image_path_or_pil
            
        results = self.reader.readtext(image_np)
        # results format: [([[x,y], [x,y], [x,y], [x,y]], text, confidence), ...]
        return results

    def parse(self, image):
        """
        Parses document into structured JSON using rules.
        """
        ocr_results = self.apply_ocr(image)
        
        # Simple extraction logic:
        # 1. Look for text patterns like "Key:"
        # 2. Find the closest text to its right or directly below it.
        
        extracted_data = {}
        
        for i, (bbox, text, prob) in enumerate(ocr_results):
            # Clean text
            clean_text = text.strip()
            
            # Check if this looks like a key (e.g., ends with colon or matches common header)
            if clean_text.endswith(":") or any(k in clean_text.lower() for k in ["date", "invoice", "total", "amount"]):
                key = clean_text.rstrip(":").strip()
                
                # Search for value in subsequent results (simple heuristic)
                value = self._find_value_near(i, ocr_results)
                if value:
                    extracted_data[key] = value
                    
        return extracted_data

    def _find_value_near(self, key_idx, ocr_results):
        """
        Heuristic to find value near a key index.
        """
        key_bbox, key_text, _ = ocr_results[key_idx]
        # key_bbox: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        k_x2 = key_bbox[1][0]
        k_y_mid = (key_bbox[0][1] + key_bbox[2][1]) / 2
        
        # Look for the first element to the right
        best_candidate = None
        min_dist = float('inf')
        
        for j, (v_bbox, v_text, _) in enumerate(ocr_results):
            if j == key_idx: continue
            
            v_x1 = v_bbox[0][0]
            v_y_mid = (v_bbox[0][1] + v_bbox[2][1]) / 2
            
            # Horizontal proximity check
            if v_x1 > k_x2 and abs(v_y_mid - k_y_mid) < 20: # 20px vertical tolerance
                dist = v_x1 - k_x2
                if dist < min_dist:
                    min_dist = dist
                    best_candidate = v_text
                    
        return best_candidate

if __name__ == "__main__":
    # Test with a dummy image if needed
    pass
