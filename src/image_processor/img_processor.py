import numpy as np
import cv2
from matplotlib import pyplot as plt

class ImageProcessor:
    def __init__(self):
        self.char_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def preprocess_and_segment_image(self, filepath):
        img = cv2.imread(filepath)

        # Display the original image of my handwriting
        plt.figure(figsize=(8, 8))
        plt.title("Ảnh gốc")
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
        plt.axis('off')
        plt.show()

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Apply adaptive threshold
        binary_image = cv2.adaptiveThreshold(blur_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 25)

        # Connected component labeling
        num_labels, labels_im = cv2.connectedComponents(binary_image, connectivity=8)

        chars = []
        bounding_boxes = []

        for label in range(1, num_labels):
            mask = (labels_im == label).astype(np.uint8) * 255
            x, y, w, h = cv2.boundingRect(mask)

            if w < 10 or h < 10:  # Ignore noise
                continue

            char = binary_image[y:y + h, x:x + w]
            resized_char = cv2.resize(char, (18, 18))
            padded_char = np.pad(resized_char, ((5, 5), (5, 5)), mode='constant', constant_values=0)

            chars.append(padded_char)
            bounding_boxes.append((x, y, w, h))

        combined = list(zip(bounding_boxes, chars))  # Fix: use chars instead of char
        sorted_combined = sorted(combined, key=lambda item: item[0][0])
        sorted_chars = [item[1] for item in sorted_combined]

        # Display sorted characters
        fig, axs = plt.subplots(1, len(sorted_chars), figsize=(15, 5))
        for i, char_img in enumerate(sorted_chars):
            axs[i].imshow(char_img, cmap='gray')
            axs[i].axis('off')
        plt.show()

        return sorted_chars

    def predict_chars(self, model, chars):
        chars_batch = np.stack(chars)
        model_pred = model.predict(chars_batch)
        pred_labels = np.argmax(model_pred, axis=1)
        pred = [self.char_labels[label] for label in pred_labels]
        return pred