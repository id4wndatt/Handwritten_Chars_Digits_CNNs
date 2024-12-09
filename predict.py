import numpy as np
import cv2
import tensorflow as tf
from matplotlib import pyplot as plt

# Định nghĩa nhãn ký tự
char_labels = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'


def preprocess_and_segment_image(filepath):
    img = cv2.imread(filepath)

    plt.figure(figsize=(8, 8))
    plt.title("Ảnh gốc")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    binary_image = cv2.adaptiveThreshold(blur_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 25)
    num_labels, labels_im = cv2.connectedComponents(binary_image, connectivity=8)

    chars = []
    bounding_boxes = []

    for label in range(1, num_labels):
        mask = (labels_im == label).astype(np.uint8) * 255
        x, y, w, h = cv2.boundingRect(mask)

        if w < 10 or h < 10:  # loại bỏ nhiễu
            continue

        char = binary_image[y:y + h, x:x + w]
        resized_char = cv2.resize(char, (18, 18))
        padded_char = np.pad(resized_char, ((5, 5), (5, 5)), mode='constant', constant_values=0)

        chars.append(padded_char)
        bounding_boxes.append((x, y, w, h))

    combined = list(zip(bounding_boxes, chars))
    sorted_combined = sorted(combined, key=lambda item: item[0][0])
    sorted_chars = [item[1] for item in sorted_combined]

    # Hiển thị chữ cái được sắp xếp
    fig, axs = plt.subplots(1, len(sorted_chars), figsize=(15, 5))
    for i, char_img in enumerate(sorted_chars):
        axs[i].imshow(char_img, cmap='gray')
        axs[i].axis('off')
    plt.show()

    return sorted_chars


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def predict_chars(model, chars):
    chars_batch = np.stack(chars)
    chars_batch = chars_batch.astype("float32") / 255  # Chuẩn hóa dữ liệu
    chars_batch = chars_batch.reshape(-1, 28, 28, 1)  # Đảm bảo dữ liệu có hình dạng (28, 28, 1)

    model_pred = model.predict(chars_batch)
    pred_labels = np.argmax(model_pred, axis=1)
    pred = [char_labels[label] for label in pred_labels]
    return pred


def main(filepath):
    model = load_model("model_result/emnist_model.keras")
    sorted_chars = preprocess_and_segment_image(filepath)
    predictions = predict_chars(model, sorted_chars)

    print("Dự đoán ký tự:", predictions)


# Gọi hàm chính với đường dẫn đến ảnh
main("test/test1.png")
main("test/test2.png")
main("test/test3.png")
main("test/test4.png")