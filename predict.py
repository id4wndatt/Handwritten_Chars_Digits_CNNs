import numpy as np
import cv2
import tensorflow as tf


class PredictImage:
    def __init__(self, model_path):
        # Load the pre-trained model
        self.model = tf.keras.models.load_model(model_path)
        self.char_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    def preprocess_img(self, filepath):
        # Preprocess an input image for prediction
        img = cv2.imread(filepath)
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
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
            chars.append(padded_char.reshape(28, 28, 1) / 255)  # Normalize and reshape
            bounding_boxes.append((x, y, w, h))  # Store bounding box

        # Sort characters based on their x-coordinate
        sorted_indices = sorted(range(len(bounding_boxes)), key=lambda i: bounding_boxes[i][0])
        sorted_chars = [chars[i] for i in sorted_indices]

        return sorted_chars

    def predict_chars(self, chars):
        if not chars:
            return []  # Return empty if no chars

        chars_batch = np.stack(chars)
        model_pred = self.model.predict(chars_batch)
        pred_labels = np.argmax(model_pred, axis=1)
        pred = [self.char_labels[label] for label in pred_labels]
        return pred


if __name__ == "__main__":
    pred_model = PredictImage('model_result/model.h5')
    chars = pred_model.preprocess_img('test/Screenshot 2024-11-21 at 8.27.24 PM.png')
    preds = pred_model.predict_chars(chars)
    print("Dự đoán:", preds)