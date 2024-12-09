import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

maps = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
    5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "A", 11: "B", 12: "C", 13: "D", 14: "E",
    15: "F", 16: "G", 17: "H", 18: "I", 19: "J",
    20: "K", 21: "L", 22: "M", 23: "N", 24: "O",
    25: "P", 26: "Q", 27: "R", 28: "S", 29: "T",
    30: "U", 31: "V", 32: "W", 33: "X", 34: "Y",
    35: "Z", 36: "a", 37: "b", 38: "d", 39: "e",
    40: "f", 41: "g", 42: "h", 43: "n", 44: "q",
    45: "r", 46: "t"
}

def build_CNNmodel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(47, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def evaluate_model(model, x, y):
    y_pred = model.predict(x)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y, axis=1)
    # In báo cáo phân loại
    print("Báo cáo độ đo:\n", classification_report(y_true_classes, y_pred_classes, target_names=list(maps.values())))

    # Vẽ ma trận nhầm lẫn
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=list(maps.values()), yticklabels=list(maps.values()))
    plt.ylabel('Thực tế')
    plt.xlabel('Đã dự đoán')
    plt.title('Ma trận Confusion')
    plt.show()