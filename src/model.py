import tensorflow as tf
NUM_CLASSES = 42
IMAGE_SIZE = (224, 224)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(
    128,
    activation="relu"
)(x)
outputs = tf.keras.layers.Dense(
    NUM_CLASSES,
    activation="softmax"
)(x)
model_transfer = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
)
model_transfer.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model_transfer.summary()
print("\nMobileNetV2 model created successfully!")
