import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Загрузка сохраненной модели
model = load_model('model.h5')

# Список путей к изображениям
img_paths = ['two.jpg', 'three.jpg', 'four.png']

for img_path in img_paths:
    # Загрузка исходной картинки
    img = image.load_img(img_path)

    # Загрузка и предобработка изображения
    processed_img = image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
    x = image.img_to_array(processed_img)
    x = np.expand_dims(x, axis=0)
    x /= 255.0

    # Предсказание цифры на изображении
    predictions = model.predict(x)
    predicted_digit = np.argmax(predictions[0])

    # Вывод исходной картинки и предполагаемой цифры
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Исходная картинка")

    plt.subplot(1, 2, 2)
    plt.imshow(processed_img, cmap='gray')
    plt.axis('off')
    plt.title("Предполагаемая цифра: {}".format(predicted_digit))

    plt.tight_layout()
    plt.show()
