import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
MODEL_PATH = "models/best_mobilenetv2_food_model.keras"
TEST_DIR = "prepared_dataset/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
print("Loading MobileNetV2 model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)
class_names = test_dataset.class_names
print("\nFood Classes:")
print(class_names)
print("\nEvaluating MobileNetV2 model...")
loss, accuracy = model.evaluate(test_dataset)
print("\nTEST RESULTS")
print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(accuracy * 100, 2), "%")
y_true = np.concatenate([
    labels.numpy() for images, labels in test_dataset
])
predictions = model.predict(test_dataset)
y_pred = np.argmax(predictions, axis=1)
print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
))
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(24, 20))
sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("MobileNetV2 Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.savefig(
    "models/mobilenetv2_confusion_matrix.png",
    dpi=300
)
plt.show()
print("\nMobileNetV2 evaluation completed successfully!")
