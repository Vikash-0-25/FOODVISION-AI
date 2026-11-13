import os
import numpy as np
import tensorflow as tf
from load import train_ds, val_ds, class_names
from model import model_transfer
from sklearn.utils.class_weight import compute_class_weight
labels = []
for class_index, class_name in enumerate(class_names):
    class_path = os.path.join("prepared_dataset", "train", class_name)
    image_count = len(os.listdir(class_path))
    labels.extend([class_index] * image_count)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=np.array(labels)
)
class_weight_dict = dict(enumerate(class_weights))
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/best_mobilenetv2_food_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2
)
history = model_transfer.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weight_dict,
    callbacks=[checkpoint, early_stopping, reduce_lr]
)
model_transfer.save("models/mobilenetv2_food_model.keras")
print("\nMobileNetV2 training completed successfully!")
print("Best model: models/best_mobilenetv2_food_model.keras")
print("Final model: models/mobilenetv2_food_model.keras")
