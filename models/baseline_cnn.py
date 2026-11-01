import tensorflow as tf
from tensorflow.keras import layers, Model


def basic_block(x):
    """Reference CNN basic block."""
    x = layers.Conv2D(12, (3, 3), activation="elu")(x)
    x = layers.Conv2D(12, (3, 3), activation="elu")(x)
    x = layers.Conv2D(12, (3, 3), activation=None)(x)
    x = layers.BatchNormalization(momentum=0.99, epsilon=0.001)(x)
    x = layers.ELU()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    return x


def build_baseline():
    inputs = layers.Input(shape=(224, 224, 3))

    x = basic_block(inputs)
    x = layers.MaxPooling2D((5, 5))(x)
    x = basic_block(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(10, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    return Model(inputs=inputs, outputs=outputs, name="baseline_cnn")


if __name__ == "__main__":
    model = build_baseline()
    model.summary()
    print("\nTotal parameters:", model.count_params())
