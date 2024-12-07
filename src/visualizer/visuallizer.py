import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class Visualizer:
    @staticmethod
    def visualize_pred(test_images, test_labels, preds, num_images=100):
        fig, axes = plt.subplots(10, 10, figsize=(15, 15))
        axes = axes.ravel()

        for i in range(num_images):
            ax = axes[i]
            ax.imshow(test_images[i].reshape(28, 28), cmap='gray')
            pred_char = preds[i]
            true_char = chr(test_labels[i] + 65)

            color = 'green' if pred_char == true_char else 'red'
            ax.set_title(f"Pred: {pred_char} | True: {true_char}", color=color)
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(test_labels, pred_labels):
        cm = confusion_matrix(test_labels, pred_labels)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[chr(i + 65) for i in range(26)],
                    yticklabels=[chr(i + 65) for i in range(26)])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Ma trận Confusion')
        plt.show()