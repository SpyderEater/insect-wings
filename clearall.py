import cv2
import os

input_folders = ['type1', 'type2']

def process_wing_structure(img):
    """Обробка зображення: виділення жилок"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 25, 5
    )
    
    return cv2.bitwise_not(thresh)

for folder_name in input_folders:
    output_folder = f"{folder_name}_processed"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Створено папку: {output_folder}")

    if not os.path.exists(folder_name):
        print(f"Помилка: Папка {folder_name} не знайдена!")
        continue

    print(f"Початок обробки папки {folder_name}...")
    
    for filename in os.listdir(folder_name):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(folder_name, filename)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
            
            processed_img = process_wing_structure(img)
            
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, processed_img)

print("\nВсі зображення оброблені та розкладені по папках '_processed'.")