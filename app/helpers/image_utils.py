from PIL import Image, ImageEnhance
from app.core.config import OCR_MAX_IMAGE_SIZE

def preprocess_image(image: Image.Image) -> Image.Image:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    max_dim = max(image.size)
    if max_dim > OCR_MAX_IMAGE_SIZE:
        ratio = OCR_MAX_IMAGE_SIZE / max_dim
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)
    return image