import pytesseract
from PIL import Image
import pandas as pd
import os

class TesseractEngine:
    def __init__(self, tesseract_cmd=None):
        """
        Initializes the Tesseract OCR engine.
        On Windows, you might need to provide the path: r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
    def extract_text(self, image_path):
        """
        Simplified text extraction.
        """
        image = Image.open(image_path)
        return pytesseract.image_to_string(image)

    def extract_data(self, image_path):
        """
        Returns a DataFrame with word-level coordinates and confidence.
        """
        image = Image.open(image_path)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
        # Clean up: remove empty strings/low confidence
        data = data[data.text.notna()]
        data = data[data.text.str.strip() != ""]
        return data

# Singleton-like access
def get_ocr_engine(tesseract_path=None):
    return TesseractEngine(tesseract_cmd=tesseract_path)
