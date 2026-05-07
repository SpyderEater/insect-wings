import numpy as np

def process_image_binary(item, root_path, radius=2, threshold=140, is_debug=False):
    import image_processor

    if item.status != "preprocess_binary":
        raise ValueError(f"Помилка: {item.relative_path} має статус {item.status}")

    img_in = item.pixels.astype(np.uint8)
    img_out = img_in.copy()
    
    image_processor.process_image_cpp(img_in, img_out, radius, threshold)

    item.pixels = img_out
    item.status = "processed_binary"
    return item
