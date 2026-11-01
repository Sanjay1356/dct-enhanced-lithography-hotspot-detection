import tensorflow as tf


def load_spatial_image(path):
    """
    Load an ICCAD lithography hotspot image using the
    reference baseline preprocessing.

    Input:
        path: Path to a PNG image.

    Output:
        float32 tensor of shape (224, 224, 3)
        with values in [0, 1].
    """
    image = tf.keras.utils.load_img(
        path,
        target_size=(224, 224)
    )

    image = tf.keras.utils.img_to_array(image)
    image = image / 255.0

    return image
