import re
import os
import json
from models.ocr_engine import get_ocr_engine

class RuleBasedPipeline:
    def __init__(self, tesseract_path=None):
        self.engine = get_ocr_engine(tesseract_path)
        
        # Regex Patterns
        self.patterns = {
            "date": r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
            "invoice_number": r"(?i)invoice\s*(?:no|number|#)[\s:]*([A-Z0-9-]+)",
            "total_amount": r"(?i)(?:total|amount|due)[\s:]*([$€£]?\s*\d+[.,]\d{2})",
            "email": r"[\w.-]+@[\w.-]+\.[a-zA-Z]{2,}"
        }

    def process(self, image_path):
        """
        Executes the full pipeline: OCR -> Regex -> Heuristics -> JSON
        """
        # 1. Structured Data Extraction
        data = self.engine.extract_data(image_path)
        full_text = " ".join(data.text.astype(str).tolist())
        
        extracted = {}
        
        # 2. Regex Extraction (Global)
        for key, pattern in self.patterns.items():
            match = re.search(pattern, full_text)
            if match:
                # Use the first group if it exists, else the full match
                extracted[key] = match.group(1) if match.groups() else match.group(0)

        # 3. Heuristic Extraction (Proximal Search)
        # Find values to the right of specific keys
        keywords = ["seller", "buyer", "tax", "subtotal"]
        for kw in keywords:
            val = self._find_value_to_right(kw, data)
            if val:
                extracted[kw] = val
                
        return extracted

    def _find_value_to_right(self, keyword, df, x_threshold=200):
        """
        Finds text to the right of a keyword within the same line (approx).
        """
        # Find keyword rows
        kw_rows = df[df.text.str.contains(keyword, case=False, na=False)]
        if kw_rows.empty:
            return None
            
        # Take the first occurrence
        kw_row = kw_rows.iloc[0]
        k_x_end = kw_row.left + kw_row.width
        k_y_mid = kw_row.top + (kw_row.height / 2)
        
        # Search for candidates to the right
        candidates = df[
            (df.left > k_x_end) & 
            (df.left < k_x_end + x_threshold) &
            (abs(df.top + (df.height / 2) - k_y_mid) < 15) # Vertical tolerance
        ]
        
        if not candidates.empty:
            # Return the first word or join multiple
            return " ".join(candidates.text.astype(str).tolist())
            
        return None

if __name__ == "__main__":
    # Internal test logic
    pass
