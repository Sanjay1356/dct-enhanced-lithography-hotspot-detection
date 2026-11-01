import tensorflow as tf


def load_dct_image(path):
    image = tf.keras.utils.load_img(
        path,
        target_size=(224, 224),
        color_mode="grayscale",
    )

    image = tf.keras.utils.img_to_array(image)
    image = image / 255.0

    # 2-D orthonormal DCT
    dct = tf.signal.dct(image[:, :, 0], type=2, norm="ortho")

    # Apply DCT along the second dimension as well
    dct = tf.transpose(
        tf.signal.dct(
            tf.transpose(dct),
            type=2,
            norm="ortho",
        )
    )

    # Magnitude compression
    dct = tf.math.abs(dct)
    dct = tf.math.log1p(dct)

    # Per-image min-max normalization
    dct_min = tf.reduce_min(dct)
    dct_max = tf.reduce_max(dct)

    dct = (dct - dct_min) / (
        dct_max - dct_min + 1e-8
    )

    # Replicate to 3 channels to match the baseline CNN
    dct = tf.stack([dct, dct, dct], axis=-1)

    return dct
