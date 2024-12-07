import tensorflow as tf


def create_cnn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(26, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


class CNNModel:
    def __init__(self):
        self.model = create_cnn_model()

    def train(self, train_images, train_labels, val_images, val_labels, epochs=5):
        self.model.fit(train_images, train_labels, epochs=epochs, validation_data=(val_images, val_labels))

    def evaluate(self, test_images, test_labels):
        val_loss, val_acc = self.model.evaluate(test_images, test_labels)
        print('Test accuracy:', val_acc)
        return val_loss, val_acc

    def save_model(self, path):
        self.model.save(path)
        print(f"Lưu model tại {path}")

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        print(f"Load model từ {path}")