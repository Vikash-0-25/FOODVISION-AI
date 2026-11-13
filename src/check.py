import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
train_dir = "prepared_dataset/train"
test_dir = "prepared_dataset/test"
class_names = sorted([
    name for name in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, name))
])
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(224, 224),
    batch_size=32,
    class_names=class_names,
    shuffle=False
)
model = tf.keras.models.load_model(
    "models/best_food_model.keras"
)
print("\nEvaluating CNN model...")
loss, accuracy = model.evaluate(test_ds)
print("\nTEST RESULTS")
print("----------------------------")
print("Test Loss:", round(loss, 4))
print("Test Accuracy:", round(accuracy * 100, 2), "%")
y_true = []
y_pred = []
for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)

    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))
report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    zero_division=0
)
print("\nCLASSIFICATION REPORT")
print("----------------------------")
print(report)
cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(class_names))
)
os.makedirs("models", exist_ok=True)
with open("models/evaluation_report.txt", "w") as file:
    file.write("Test Accuracy: " + str(accuracy * 100) + "%\n")
    file.write(report)
np.savetxt(
    "models/confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d"
)
print("\nEvaluation completed successfully!")
print("Results saved inside models folder.")
