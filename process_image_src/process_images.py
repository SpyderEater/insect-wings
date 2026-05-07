import numpy as np
import image_processor

def process_image_binary(item, root_path, radius=2, threshold=100, is_debug=False):
    if item.status != "preprocess_binary":
        raise ValueError(f"Помилка: {item.relative_path} має статус {item.status}")

    img_out = np.empty_like(item.pixels)
    
    image_processor.process_image_cpp(item.pixels, img_out, radius, threshold)

    item.pixels = img_out
    item.status = "processed_binary"
    return item
