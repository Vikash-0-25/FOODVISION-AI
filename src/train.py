import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from load import train_ds, val_ds, class_names
from sam import model
os.makedirs("models", exist_ok=True)
class_counts = []
for class_name in class_names:
    folder = os.path.join("prepared_dataset", "train", class_name)
    count = len([
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])
    class_counts.append(count)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=np.repeat(
        np.arange(len(class_names)),
        class_counts
    )
)
class_weight_dict = {
    i: float(weight)
    for i, weight in enumerate(class_weights)
}
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "models/best_food_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1
)
print("\nStarting CNN model training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    class_weight=class_weight_dict,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr
    ]
)
model.save("models/food_freshness_model.keras")
print("\nTraining completed successfully!")
print("Best model: models/best_food_model.keras")
print("Final model: models/food_freshness_model.keras")
