from PIL import Image
from pdf2image import convert_from_path
import os

def load_and_convert(file_path):
    """
    Loads a file (PDF or Image) and returns a list of PIL Images.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return convert_from_path(file_path)
    else:
        return [Image.open(file_path).convert("RGB")]

def resize_for_model(image, target_size=(1000, 1000)):
    """
    Resizes image and returns the new image and scaling factors.
    """
    w, h = image.size
    resized_img = image.resize(target_size, Image.BILINEAR)
    return resized_img, (target_size[0] / w, target_size[1] / h)

def save_temp_image(image, prefix="tmp"):
    """
    Saves image to a temporary file and returns path.
    """
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".png", prefix=prefix)
    image.save(path)
    os.close(fd)
    return path
