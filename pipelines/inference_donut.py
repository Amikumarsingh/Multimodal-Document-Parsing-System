from models.donut import DonutWrapper
from PIL import Image

class DonutPipeline:
    def __init__(self, model_path="naver-clova-ix/donut-base"):
        self.model_wrapper = DonutWrapper(model_checkpoint=model_path)
        
    def process(self, image_input):
        """
        Runs Donut generative parsing.
        image_input can be a file path or PIL Image.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        else:
            image = image_input
            
        result = self.model_wrapper.predict(image)
        return result
