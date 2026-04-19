from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Processor
import torch

class LayoutLMV3Model:
    def __init__(self, model_checkpoint="microsoft/layoutlmv3-base", num_labels=7):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = LayoutLMv3Processor.from_pretrained(model_checkpoint, apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(
            model_checkpoint, 
            num_labels=num_labels
        ).to(self.device)
        
    def forward(self, encoding):
        """
        Standard forward pass for inference.
        """
        # Move to device
        inputs = {k: v.to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        return outputs.logits

    def save(self, path):
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
