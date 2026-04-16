from PIL import Image
import numpy as np

class TempItem:
    pass

def debug_log_pixel_avg(log_file, y, x, avg):
    if avg != 0:
        log_file.write(f"P({y},{x}) Avg: {avg:.2f}\n")

def process_wrapper(args):
    relative_path, root, radius, threshold, is_debug = args

    input_path = root / "input_images" / relative_path
    #img = Image.open(input_path).convert("L")

    pixels = np.array(img)
    print("DEBUG SHAPE:", pixels.shape, pixels.dtype)
    
    item = TempItem()
    item.pixels = pixels
    item.relative_path = relative_path
    item.status = "preprocess_binary"
    
    result = process_image_binary(item, root, radius, threshold, is_debug)
    
    return (result.pixels, result.relative_path)

def process_image_binary(item, root_path, radius=1, threshold=100, is_debug=False):
    if item.status != "preprocess_binary":
        raise ValueError(f"Помилка: {item.relative_path} має статус {item.status}")
    
    print("DEBUG SHAPE:", item.pixels.shape, item.pixels.dtype)

    # img = item.pixels
    img = item.pixels.astype(np.int32)

    height, width = img.shape
    new_img = img.copy()
    
    side = 2 * radius + 1
    total_pixels = side * side
    median_idx = total_pixels // 2
    
    log_file = None
    if is_debug:
        log_dir = root_path / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file_path = log_dir / item.relative_path.with_suffix('.txt')
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_file_path, "w")

    try:
        for y in range(radius, height - radius):
            for x in range(radius, width - radius):
                
                window = []
                for dy in range(-radius, radius + 1):
                    for dx in range(-radius, radius + 1):
                        window.append(img[y + dy, x + dx])
                
                avg_brightness = sum(int(p) for p in window) / total_pixels
                
                if avg_brightness > threshold:
                    new_img[y, x] = 255
                else:
                    window.sort()
                    pixel_result = window[median_idx]
                    
                    if pixel_result == 0 and is_debug:
                        debug_log_pixel_avg(log_file, y, x, avg_brightness)
                    
                    new_img[y, x] = pixel_result
    finally:
        if log_file:
            log_file.close()

    # item.pixels = new_img
    item.pixels = new_img.astype(np.uint8)

    item.status = "processed_binary"
    
    return item
