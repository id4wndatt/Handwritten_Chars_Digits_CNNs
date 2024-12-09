from src.data_loader.data_loader import load_data, create_dataset
from src.model.model import build_CNNmodel, evaluate_model
from src.utils.utils import display_data, rotate_data

import numpy as np
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = load_data()
rotated_x_train = rotate_data(x_train)
rotated_x_test = rotate_data(x_test)

# Chuyển đổi định dạng cho TensorFlow
dataset = create_dataset(rotated_x_train, y_train)
test_dataset = create_dataset(rotated_x_test, y_test)

model = build_CNNmodel()
callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
model.fit(dataset, validation_data=test_dataset, epochs=20, callbacks=callbacks, verbose=2)

print("Đánh giá Test Data:")
evaluate_model(model, rotated_x_test, y_test)

print("Đánh giá Train Data:")
evaluate_model(model, rotated_x_train, y_train)

# Dự đoán một số mẫu ngẫu nhiên
random_sample_val = np.random.choice(rotated_x_test.shape[0], 9, replace=False)
random_samples = rotated_x_test[random_sample_val]
true_labels = y_test[random_sample_val]

predictions = model.predict(random_samples)
display_data(random_samples, true_labels, predictions)

model.save("model_result/emnist_model.keras")