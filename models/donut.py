try:
    from transformers import DonutForConditionalGeneration
except ImportError:
    from transformers import VisionEncoderDecoderModel as DonutForConditionalGeneration
from transformers import DonutProcessor
import torch
from PIL import Image
import re

class DonutWrapper:
    def __init__(self, model_checkpoint="naver-clova-ix/donut-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = DonutProcessor.from_pretrained(model_checkpoint)
        self.model = DonutForConditionalGeneration.from_pretrained(model_checkpoint).to(self.device)
        
    def predict(self, image: Image.Image):
        """
        Generates structured JSON directly from image.
        """
        # Prepare decoder inputs
        task_prompt = "<s_funsd>"
        decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        outputs = self.model.generate(
            pixel_values.to(self.device),
            decoder_input_ids=decoder_input_ids.to(self.device),
            max_length=self.model.config.decoder.max_position_embeddings,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
        
        prediction = self.processor.batch_decode(outputs.sequences)[0]
        prediction = self.processor.token2json(prediction)
        
        return prediction

def clean_donut_output(output_json):
    """
    Cleans up the raw JSON from Donut.
    """
    # Simple pass-through for now
    return output_json
