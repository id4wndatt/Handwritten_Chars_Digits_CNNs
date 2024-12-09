import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.model.model import maps


def display_data(x, y, predictions=None): # Hiển thị dữ liệu với nhãn thực tế và nhãn dự đoán.
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x[i], cmap="gray")
        title = f'True: {maps[np.argmax(y[i])]}'
        if predictions is not None:
            title += f'\nPred: {maps[np.argmax(predictions[i])]}'
        plt.title(title)
        plt.axis("off")
    plt.show()

def rotate_data(x): #Xoay và lật ngược ảnh
    return np.array([np.array(Image.fromarray(img).rotate(90).transpose(Image.FLIP_TOP_BOTTOM)) for img in x])