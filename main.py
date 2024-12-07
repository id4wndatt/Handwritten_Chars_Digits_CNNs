import numpy as np

from src.data_loader.data_loader import DataLoader
from src.image_processor.img_processor import ImageProcessor
from src.model.model import CNNModel
from src.visualizer.visuallizer import Visualizer

if __name__ == "__main__":
    # Load and preprocess data
    data_loader = DataLoader('data/A_Z Handwritten Data.csv')
    train_images, test_images, train_labels, test_labels = data_loader.preprocess_data()

    # Create and train the model
    cnn_model = CNNModel()
    cnn_model.train(train_images, train_labels, test_images, test_labels, epochs=5)
    cnn_model.save_model('model_result/model.h5')

    # Evaluate the model
    val_loss, val_acc = cnn_model.evaluate(test_images, test_labels)

    # Make predictions
    pred = cnn_model.model.predict(test_images)
    pred_labels = np.argmax(pred, axis=1)

    # Visualize predictions
    vs = Visualizer()
    vs.visualize_pred(test_images, test_labels, pred_labels)
    vs.plot_confusion_matrix(test_labels, pred_labels)

    # Process and predict custom images
    image_processor = ImageProcessor()
    test_image_path=(r'test/Screenshot 2024-11-21 at 8.27.24 PM.png')
    chars = image_processor.preprocess_and_segment_image(test_image_path)
    preds = image_processor.predict_chars(cnn_model.model, chars)
    print(f"Dự đoán cho {test_image_path}: {preds}")