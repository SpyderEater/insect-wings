import cv2
import matplotlib.pyplot as plt


image_path = "Lestes sponsa France 1-1.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"Помилка: Не вдалося завантажити зображення за шляхом {image_path}")
    exit()


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 5)


final_img = cv2.bitwise_not(thresh)

print("Оригінальне зображення:")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Оригінальне зображення")
plt.axis('off')
plt.show()
print("Результат з виділеною структурою жилок:")
plt.imshow(final_img, cmap='gray')
plt.title("Результат з виділеною структурою жилок")
plt.axis('off')
plt.show()
cv2.imwrite("wing_structure_final.png", final_img)
