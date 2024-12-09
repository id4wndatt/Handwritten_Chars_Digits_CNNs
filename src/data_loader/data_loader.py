import os
import pandas as pd
import tensorflow as tf

datadir = "data"

def import_data(): # Nhập dữ liệu từ file CSV
    train = pd.read_csv(os.path.join(datadir, "emnist-balanced-train.csv"), header=None)
    test = pd.read_csv(os.path.join(datadir, "emnist-balanced-test.csv"), header=None)

    x_train = train.iloc[:, 1:].values.reshape(-1, 28, 28)
    y_train = train.iloc[:, 0].values
    x_test = test.iloc[:, 1:].values.reshape(-1, 28, 28)
    y_test = test.iloc[:, 0].values

    return (x_train, y_train), (x_test, y_test)

def load_data(): # Tải và chuẩn hóa dữ liệu
    (x_train, y_train), (x_test, y_test) = import_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    y_train = tf.keras.utils.to_categorical(y_train, 47)
    y_test = tf.keras.utils.to_categorical(y_test, 47)
    return (x_train, y_train), (x_test, y_test)

def create_dataset(x, y, batch_size=32):
    x = x.reshape(-1, 28, 28, 1) # Thêm chiều kênh
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(10000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE) # Đảm bảo dữ liệu có hình dạng (28, 28, 1)
    return dataset